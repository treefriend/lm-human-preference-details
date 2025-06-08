# 下面为每一行添加详细的中文注释，涵盖理论基础和工程实现，适合初学者理解
import functools  # functools 提供高阶函数工具，如 partial 用于函数柯里化
import os  # os 用于操作系统相关操作，如路径处理
import random  # random 用于设置随机种子，保证实验可复现
import time  # time 用于计时和生成时间戳
from dataclasses import asdict, dataclass, field  # dataclasses 简化数据结构定义，常用于实验参数管理
from types import SimpleNamespace  # SimpleNamespace 可快速创建可动态赋值的对象
from typing import List, Optional  # 类型注解，提升代码可读性和类型检查

import numpy as np  # numpy 是常用的数值计算库
import pandas as pd  # pandas 用于数据分析和表格展示
import torch  # PyTorch 主库，支持张量运算和深度学习
import torch.nn as nn  # PyTorch 的神经网络模块
import torch.nn.functional as F  # PyTorch 的常用函数，如 softmax、loss 等
import torch.optim as optim  # PyTorch 的优化器模块
import tyro  # tyro 用于命令行参数解析，简化实验参数管理
from accelerate import Accelerator  # accelerate 用于分布式训练和多设备管理
from accelerate.state import AcceleratorState  # AcceleratorState 用于获取分布式训练状态
from datasets import load_dataset  # HuggingFace datasets，用于加载标准数据集
from rich.console import Console  # rich 用于美观的终端输出
from rich.pretty import pprint  # rich 的 pprint 用于结构化美观打印
from rich.table import Table  # rich 的 Table 用于表格展示
from torch import Tensor, optim  # Tensor 类型和优化器
from torch.optim.optimizer import (  # 优化器底层实现细节
    _dispatch_sqrt,
    _get_value,
    _use_grad_for_differentiable,
)
from torch.utils.data import DataLoader  # PyTorch 的数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志记录器，用于可视化训练过程
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # HuggingFace 的模型和分词器


@dataclass  # 使用dataclass简化类定义，自动生成__init__等方法
class AdaptiveKLParams:
    target: float = 6.0  # 目标KL散度，控制policy和reference的相似度，防止policy偏离太远
    horizon: int = 10000  # 自适应KL调整的时间窗口（episode数），影响KL系数调整速度


@dataclass  # 奖励模型相关超参数
class RewardHParams:
    kl_coef: float = 0.15  # KL散度系数，控制policy和reference的距离惩罚强度
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)  # 是否使用自适应KL，及其参数
    trained_model: Optional[str] = "models/reward/pytorch_model.bin"  # 预训练reward模型的路径
    label_dataset: tyro.conf.Suppress[Optional[str]] = None  # 标注数据集路径，通常用于监督微调


@dataclass  # PPO算法相关超参数
class PpoHParams:
    total_episodes: int = 1000000  # 总训练episode数，决定训练轮数
    local_batch_size: int = 64  # 单进程/单卡的batch size，影响显存占用和采样效率
    local_mini_batch_size: tyro.conf.Suppress[int] = None  # 单进程/单卡的minibatch size，供梯度累积和分布式用
    batch_size: tyro.conf.Suppress[int] = None  # 全局batch size（所有进程/卡总和），自动推导
    mini_batch_size: tyro.conf.Suppress[int] = None  # 全局minibatch size，自动推导
    gradient_accumulation_steps: int = 1  # 梯度累积步数，提升大batch训练能力
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None  # 单进程/卡的micro batch size，供梯度累积用
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None  # 分布式训练的进程/卡总数，自动推导
    batch_size: tyro.conf.Suppress[int] = None  # 冗余定义，实际由上面推导
    minibatch_size: tyro.conf.Suppress[int] = None  # 冗余定义，实际由上面推导
    num_updates: tyro.conf.Suppress[int] = None  # 总更新步数，自动推导
    nminibatches: int = 1  # 每个epoch的minibatch数，影响shuffle和优化器步数
    noptepochs: int = 4  # 每个采样批次的PPO优化epoch数，提升样本利用率
    lr: float = 0.00001  # 学习率，控制参数更新步幅
    eps: float = 1e-5  # Adam优化器的epsilon，提升数值稳定性
    vf_coef: float = 0.1  # value function loss的权重，PPO损失的加权项
    cliprange: float = 0.2  # PPO策略损失的裁剪范围，防止策略更新过大
    cliprange_value: float = 0.2  # value function损失的裁剪范围
    gamma: float = 1  # 折扣因子，控制未来奖励的权重，RL基础参数
    lam: float = 0.95  # GAE(lambda)的lambda参数，控制优势估计的偏差与方差权衡
    whiten_rewards: bool = True  # 是否对奖励进行归一化（白化），提升训练稳定性


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 64
    query_dataset: str = "books"

    # Response params
    response_length: int = 24

    # 采样时，在索引 after 及之后首次出现该 token 时截断响应
    truncate_token: int = 13
    # 从该索引开始检查截断 token
    truncate_after: int = 16
    # 惩罚奖励值
    penalty_reward_value: int = -1

    # LM params
    # 语言模型生成时的温度参数，控制随机性
    temperature: float = 0.7


@dataclass
class Args:
    # common args
    # 本实验的名称，取当前文件名去掉 .py 后缀
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    # 实验的随机种子，保证实验可复现
    seed: int = 1
    """seed of the experiment"""
    # 是否使用 Weights and Biases 跟踪实验
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    # Weights and Biases 项目的名称
    wandb_project_name: str = "cleanrl"
    """the wandb's project name"""
    # Weights and Biases 项目所属的实体（团队）
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    # 是否在可用时使用 CUDA 加速
    cuda: bool = True
    """Whether to use cuda if available."""
    # 本次运行的唯一名称，待填充
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""
    # 是否将保存的模型上传到 Hugging Face
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    # Hugging Face Hub 上模型仓库所属的用户或组织名称
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"

    # 预训练模型的名称
    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    # 是否使用 DeepSpeed 训练模型
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    # 打印样本输出的频率
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    # 模型保存的路径
    save_path: str = "models/policy"
    """Where to save the model"""
    # 是否使用 TensorFlow 风格的 Adam 优化器替代 PyTorch 的
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    # 任务相关的超参数
    task: TaskHParams = field(default_factory=TaskHParams)
    # 奖励模型相关的超参数
    rewards: RewardHParams = field(default_factory=RewardHParams)
    # PPO 算法相关的超参数
    ppo: PpoHParams = field(default_factory=PpoHParams)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    # 创建一个带有分隔线的表格
    table = Table(show_lines=True)
    # 为表格添加列
    for column in df.columns:
        table.add_column(column)
    # 为表格添加行
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    # 打印表格标题
    console.rule(f"[bold red]{title}")
    # 打印表格
    console.print(table)


def _single_tensor_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    # 确保没有梯度缩放和无穷大值检测
    assert grad_scale is None and found_inf is None

    # 遍历所有参数
    for i, param in enumerate(params):
        # 根据是否最大化目标调整梯度方向
        grad = grads[i] if not maximize else -grads[i]
        # 一阶矩估计
        exp_avg = exp_avgs[i]
        # 二阶矩估计
        exp_avg_sq = exp_avg_sqs[i]
        # 当前步数
        step_t = state_steps[i]
        # 更新步数
        step_t += 1
        # 衰减一阶矩估计
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # 衰减二阶矩估计
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        # 获取当前步数的数值
        step = _get_value(step_t)

        ### pytorch adam implementation:
        # bias_correction1 = 1 - beta1 ** step
        # bias_correction2 = 1 - beta2 ** step
        # step_size = lr / bias_correction1
        # bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        # param.addcdiv_(exp_avg, denom, value=-step_size)

        ### tensorflow adam implementation:
        # 计算调整后的学习率
        lr_t = lr * _dispatch_sqrt(1 - beta2**step) / (1 - beta1**step)
        # 计算分母
        denom = exp_avg_sq.sqrt().add_(eps)
        # 更新参数
        param.addcdiv_(exp_avg, denom, value=-lr_t)


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    # 使用单张量版的 Adam 优化器实现
    func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


class AdamTensorFlowStyle(optim.Adam):
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # 使用正态分布初始化权重
    torch.nn.init.normal_(layer.weight, std=std)
    # 初始化偏置为常数
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


'''
在 RLHF/PPO 训练中，通常会对 policy 和 reference policy（如 SFT 模型）之间的 KL 散度加一个惩罚项，防止 policy 偏离 reference 太远。
这个惩罚项的强度由 KL 系数（kl_coef）控制。
如果 KL 系数太小，policy 可能发散；太大，policy 学不到新东西。
自适应 KL 控制就是根据实际训练中观测到的 KL 散度，动态调整 KL 系数，使 KL 散度保持在目标值附近。
初始化时保存当前 KL 系数和目标参数。
每次调用 update(current, n_steps)，会根据当前观测到的 KL（current）和目标 KL（target），调整 KL 系数（self.value）。
调整策略是：如果当前 KL 比目标大，就增大 KL 系数（惩罚更强）；如果当前 KL 比目标小，就减小 KL 系数（惩罚更弱）。
这样可以让训练过程中的 KL 散度自动收敛到目标值，提升训练稳定性。
target是一个超参数，一般经验值是 1~10，具体取值要结合模型规模、任务和实验效果调参。
如果 target 设得太小，policy 学不到新东西；太大，policy 可能发散。
可以先用默认值（如 6.0），训练时观察实际 KL 曲线和模型表现，再微调。
  kl = logprobs - ref_logprobs
  non_score_reward = -kl_ctl.value * kl
  
  在 RLHF/PPO 训练中，总的 loss 主要包括两部分：

策略损失（policy loss）：用 advantage 和 概率比值 ratio 计算，通常带有 clip。
KL penalty（KL 惩罚项）：用来惩罚 policy 和 reference policy（如 SFT 模型）之间的分布差异，防止 policy 偏离太远。
这个 penalty 会加到 reward 上，影响 advantage，从而影响最终的 policy loss。

policy参数
   ↓
logprobs
   ↓
kl = logprobs - ref_logprobs
   ↓
non_score_reward = -kl_ctl.value * kl
   ↓
rewards = non_score_reward + scores
   ↓
advantage, returns
   ↓
policy loss (含clip)
   ↓
loss.backward()  # 反向传播
'''
class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        # 初始化 KL 系数
        self.value = init_kl_coef
        # 保存超参数
        self.hparams = hparams

    def update(self, current, n_steps):
        # 获取目标 KL 散度，希望模型训练过程中KL在target附近变动
        # 如果大了，小了，都调整下
        target = self.hparams.target
        # 计算比例误差，并限制在 [-0.2, 0.2] 范围内
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        # 计算调整倍数
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        # 更新 KL 系数
        self.value *= mult


'''
对输入的张量（如 advantage 或 reward）做标准化（白化）处理，使其均值为 0，方差为 1。

为什么要这样做？
在 RLHF/PPO 等强化学习训练中，advantage 或 reward 的分布可能会有较大波动。
对 advantage 或 reward 做标准化（白化），可以让它们的分布更适合优化器训练，提高训练的稳定性和收敛速度。
这类似于深度学习中的“归一化”操作。
'''
def whiten(values, shift_mean=True):
    # 计算均值和方差，`unbiased=False` 匹配 TF `tf.nn.moments` 的设置
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    # 计算白化后的数值
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    # 如果不移动均值，则加上均值
    if not shift_mean:
        whitened += mean
    return whitened

'''
policy（策略模型）和 reference policy 都是用这个类包装的
'''
class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        # 保存语言模型主干
        self.lm_backbone = lm_backbone
        # 初始化标量头,这个“标量头”通常用来输出每个 token 的 value（状态价值），用于 PPO 算法中的 value loss 计算
        self.scalar_head = layer_init(nn.Linear(lm_backbone.config.hidden_size, 1), std=0)

    def forward(self, **kwargs):
        # 前向传播语言模型主干
        output = self.lm_backbone(**kwargs)
        # 通过标量头计算输出
        return output, self.scalar_head(output.hidden_states[-1])


'''
这是一个在语言模型主干后面加了一个“奖励头”（reward head）的模型。
主要用于 reward model（奖励模型），即对 query-response 对打分。
奖励头通常只取最后一个 token 的隐藏状态，经过线性层输出一个 reward 分数。
还带有可学习的 reward_gain 和 reward_bias，用于归一化 reward 分布。
'''
class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        # 保存语言模型主干
        self.lm_backbone = lm_backbone
        # 初始化奖励头
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        # 奖励增益参数
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # 奖励偏置参数
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs):
        # 前向传播语言模型主干
        output = self.lm_backbone(**kwargs)
        # 获取最后一层的隐藏状态
        reward_latents = output.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]
        # 获取最后一个时间步的隐藏状态
        last_reward_latents = reward_latents[:, -1, :]
        # shape: [batch_size, hidden_size]
        # 通过奖励头计算奖励
        reward = self.scalar_head(last_reward_latents)
        # shape: [batch_size, 1]
        # 应用奖励增益和偏置
        reward = self.reward_gain * reward + self.reward_bias
        return output, reward


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding."""
    # 确保输入的张量是二维的
    assert tokens.ndim == 2
    return torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    # restore padding tokens
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

'''
query_responses:
[[101, 102, 103, 0, 0],
 [201, 202, 203, 204, 0]]
 
 attention_mask:
 [[1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0]]
 
 position_ids:
 [[0, 1, 2, 2, 2],
 [0, 1, 2, 3, 3]]
 
 input_ids:
 [[101, 102, 103, 0, 0],
 [201, 202, 203, 204, 0]]
 
 [[101, 102, 103, 0, 0],
 [201, 202, 203, 204, 0]]
 
 这个函数的作用是：把 query+response 序列送入 policy 模型，得到每个 token 的 logits（概率分布）和 value（状态价值）。
这样可以用于后续的 logprobs、KL、reward、advantage 等 PPO/RLHF 训练核心计算。

query_responses 作为 input_ids 输入模型，模型会对每个 token 位置输出 logits（即预测下一个 token 的分布）。
例如，output.logits[:, i, :] 就是第 i 个 token 位置预测下一个 token 的 logits。
在 PPO 训练中，通常会用 logits 计算 logprobs，再和实际生成的 response token 做比对，得到每个 token 的 log 概率。
'''
def forward(policy, query_responses, tokenizer):
    # 1. 生成 attention_mask，标记哪些位置是有效token（非pad）
    attention_mask = query_responses != tokenizer.pad_token_id

    # 2. 生成 position_ids，记录每个token在序列中的位置（exclusive cumsum，常用于transformer位置编码）
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
 
    # 3. 复制一份输入序列，pad位置填0（防止pad token影响模型输出）
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0

    # 4. 前向传播，返回模型输出（output含logits/hidden_states等，scalar_head输出value）
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


'''
采样阶段 
    用 policy 生成 response
    计算 policy 和 reference 的 log 概率
    对 response 做截断等后处理
    用 reward model 打分
    计算 KL penalty，合成最终 reward
    对 reward 做归一化

优势估计（GAE）
    用 GAE 算法计算每个 token 的 advantage 和 returns
    对 advantage 做归一化
    
PPO 多轮优化
    多个 epoch，每次随机打乱样本
    分 minibatch、microbatch 迭代
    前向计算新策略的 logprobs、value
    计算 PPO 损失（含 clip）、value 损失
    反向传播，更新参数
    记录各种统计量和日志
'''
def train(args: Args):
    accelerator = Accelerator(gradient_accumulation_steps=args.ppo.gradient_accumulation_steps)  # 初始化Accelerator，支持分布式训练和梯度累积
    args.ppo.world_size = accelerator.num_processes  # 获取当前分布式训练的进程/卡总数
    args.ppo.batch_size = int(args.ppo.local_batch_size * args.ppo.world_size)  # 计算全局batch size
    args.ppo.minibatch_size = exact_div(args.ppo.batch_size, args.ppo.nminibatches)  # 计算全局minibatch size
    args.ppo.local_mini_batch_size = exact_div(args.ppo.local_batch_size, args.ppo.nminibatches)  # 计算单进程minibatch size
    args.ppo.local_micro_batch_size = exact_div(args.ppo.local_mini_batch_size, args.ppo.gradient_accumulation_steps)  # 计算单进程micro batch size
    if args.ppo.whiten_rewards:  # 如果需要对奖励归一化
        assert (
            args.ppo.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"  # 保证归一化时每个minibatch样本数足够
    # `per_rank_rollout_batch_size` 是每个进程的采样batch size
    # `per_rank_minibatch_size` 是每个进程的minibatch size
    args.ppo.num_updates = args.ppo.total_episodes // args.ppo.batch_size  # 计算总的更新步数

    console = Console(force_terminal=True)  # rich的Console用于美观输出
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"  # 生成唯一的实验运行名
    writer = SimpleNamespace()  # 创建一个空对象作为日志记录器（默认无操作）
    writer.add_scalar = lambda x, y, z: None  # 默认add_scalar为空操作
    writer.add_histogram = lambda x, y, z: None  # 默认add_histogram为空操作
    if accelerator.is_main_process:  # 只在主进程做日志和可视化
        if args.track:  # 如果需要用wandb跟踪实验
            import wandb  # 动态导入wandb

            wandb.init(
                project=args.wandb_project_name,  # wandb项目名
                entity=args.wandb_entity,  # wandb团队名
                sync_tensorboard=True,  # 同步tensorboard日志
                config=asdict(args),  # 记录所有超参数
                name=run_name,  # wandb运行名
                save_code=True,  # 保存代码快照
            )
            wandb.run.log_code(".")  # 上传当前目录下的代码
        writer = SummaryWriter(f"runs/{run_name}")  # 初始化TensorBoard日志记录器
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),  # 记录所有超参数到TensorBoard
        )
        pprint(args)  # 结构化打印所有参数
    device = accelerator.device  # 获取当前设备（CPU/GPU）
    local_seed = args.seed + accelerator.process_index * 100003  # 为每个进程生成唯一随机种子，保证分布式下数据不同
    random.seed(local_seed)  # 设置python随机种子
    np.random.seed(local_seed)  # 设置numpy随机种子
    torch.manual_seed(local_seed)  # 设置torch随机种子
    torch.backends.cudnn.deterministic = True  # 保证cudnn确定性，提升可复现性
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )  # 加载分词器，右侧padding，支持自定义模型
    # 手动添加pad token，但不调整模型embedding大小
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )  # 加载奖励模型（主干+奖励头）
    if args.rewards.trained_model:
        reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))  # 加载预训练reward模型参数
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")
    # 参考策略模型（不参与训练，仅用于KL惩罚）
    # reference model 通常是 policy 初始化时的快照（如 SFT 模型），或者是训练初始的 policy。
    # 训练过程中 reference model 保持不变，policy model 不断优化，KL penalty 用来约束 policy 不要偏离 reference 太远。
    ref_policy = AutoModelForCausalLMWithScalarHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    # 训练的策略模型
    '''
    相对于sft，多了两个模型，显存就大了很多
    另外，就是ppo 每个batch 都要用policy采样，生成的序列通常比sft的label长，且生成时保留很多中间状态
    对于每个token，ppo需要存储logprobs，ref_logprobs，values，Advantages，rewards，returns，而sft只需要保存logits和loss
    '''
    policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    policy.lm_backbone.generation_config.eos_token_id = (
        None  # 禁用eos和pad token，保证生成长度固定
    )
    policy.lm_backbone.generation_config.pad_token_id = None  # 禁用pad token
    # PyTorch 原生 Adam 和 TensorFlow 风格 Adam 的主要区别在于偏置校正和学习率调整公式，
    # 实际参数更新步幅略有不同。 RLHF 工程常用 TensorFlow 风格以对齐社区实现。
    # 在大多数常规任务下，两者效果接近，但在 RLHF/PPO 等对优化器敏感的场景下，可能会有收敛速度或稳定性上的差异。
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(policy.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)  # 使用TF风格Adam
    else:
        optimizer = optim.Adam(policy.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)  # 使用PyTorch原生Adam
    dataset = load_dataset("bookcorpus", split="train")  # 加载BookCorpus数据集
    dataset = dataset.shuffle(seed=local_seed)  # 打乱数据集，保证每个进程数据不同

    def process_query_data(x, base_model: str, response_length: int):  # 数据预处理函数
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return {
            "query_token": tokenizer(
                x["text"], padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
            )["input_ids"],
        }

    dataset.set_transform(
        functools.partial(process_query_data, base_model=args.base_model, response_length=args.task.response_length)
    )  # 设置数据集的transform，保证每次取出都是token id
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)  # 构建PyTorch DataLoader
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)  # 分布式封装模型、优化器、数据
    if args.deepspeed:
        import deepspeed  # 动态导入deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin  # 获取deepspeed配置
        # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.ppo.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            # "steps_per_print": 10,
            # "zero_optimization": { ... },
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)  # deepspeed封装reward模型
        reward_model.eval()  # 评估模式
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)  # deepspeed封装ref_policy
        ref_policy.eval()
    else:
        ref_policy = ref_policy.to(device)  # 将ref_policy移动到当前设备
        reward_model = reward_model.to(device)  # 将reward_model移动到当前设备

    def repeat_generator():  # 无限循环dataloader，保证每次采样都有数据
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())  # 构建可迭代数据流
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)  # 初始化KL控制器
    # 警告：即使max_new_tokens和min_new_tokens相同，生成长度也可能不同
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,  # 生成的最大新token数
        min_new_tokens=args.task.response_length,  # 生成的最小新token数
        temperature=args.task.temperature,  # 采样温度
        top_k=0.0,  # 不做top-k截断
        top_p=1.0,  # 不做top-p截断
        do_sample=True,  # 启用采样
    )

    print("===training policy===")  # 打印训练开始提示
    global_step = 0  # 全局步数计数器
    stats_shape = (args.ppo.noptepochs, args.ppo.nminibatches, args.ppo.gradient_accumulation_steps)  # 统计量shape
    approxkls_stats = torch.zeros(stats_shape, device=device)  # KL散度统计
    clipfracs_stats = torch.zeros(stats_shape, device=device)  # 策略裁剪比例统计
    pg_losses_stats = torch.zeros(stats_shape, device=device)  # 策略损失统计
    vf_losses_stats = torch.zeros(stats_shape, device=device)  # value损失统计
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)  # value裁剪比例统计
    entropies_stats = torch.zeros(stats_shape, device=device)  # 熵统计
    # ====== PPO主训练循环，每次update为一次完整的采样-优化流程 ======
    '''
    flowchart TD
    A[开始一个PPO update循环] --> B[采样一批query]
    B --> C[用policy生成response]
    C --> D[计算policy和reference的logprobs、value]
    D --> E[对response做截断处理]
    E --> F[用reward model对response打分]
    F --> G[过滤不合格response，惩罚分]
    G --> H[计算KL penalty和最终reward]
    H --> I[对reward做归一化]
    I --> J[用GAE算法计算advantage和returns]
    J --> K[PPO多轮优化（epoch、minibatch、microbatch）]
    K --> L[反向传播，更新policy参数]
    L --> M[记录日志和统计量]
    M --> N[进入下一个update循环或训练结束]
    '''
    for update in range(1, args.ppo.num_updates + 1):
        # ====== PPO主训练循环，每次update代表一次完整的采样+多轮优化 ======
        global_step += 1 * args.ppo.batch_size  # 累加全局步数
        frac = 1.0 - (update - 1.0) / args.ppo.num_updates  # 计算当前训练进度比例（用于学习率衰减）
        lrnow = frac * args.ppo.lr  # 按线性衰减调整当前学习率
        optimizer.param_groups[0]["lr"] = lrnow  # 动态设置优化器的学习率
        data = next(iter_dataloader)  # 从数据流中取出一个batch的query
        with torch.no_grad():  # 采样阶段不需要梯度，节省显存
            queries = data["query_token"].to(device)  # 获取query的token并放到设备上
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)  # 右padding转左padding，适配生成
            # === 用当前policy模型对query进行采样，生成response ===
            query_responses = generate(
                accelerator.unwrap_model(policy).lm_backbone,
                queries,
                tokenizer,
                generation_config,
            )
            # ====== 下面详细解释每个变量的含义和作用 ======
            # context_length：当前 batch 的 query 序列长度（即 prompt 长度），用于后续分割 query 和 response。
            context_length = queries.shape[1]  # 记录query长度，后续用于分割query/response
            # responses：从 query_responses 中切片得到的 response 部分（去掉前面的 query 部分），shape 为 [batch, response_length]。
            responses = query_responses[:, context_length:]

            # 先把回复拿到，然后去评估回复的logits，理论上，logits在上面generate那里也能拿，是为了梯度回归放到第二次
            # output：policy 模型的输出（transformers风格，含 logits、hidden_states 等）。
            # full_values：policy 模型 value head 的输出，shape 为 [batch, seq_len, 1]，表示每个 token 的 value 估计。
            output, full_values = forward(policy, query_responses, tokenizer)
            # values：取出 response 部分的 value，shape 为 [batch, response_length]，用于后续 GAE/advantage 计算。
            values = full_values[:, context_length - 1 : -1].squeeze(-1)
            # logits：policy 模型输出的 logits，取 response 部分，shape 为 [batch, response_length, vocab_size]。
            logits = output.logits[:, context_length - 1 : -1]
            # logits /= args.task.temperature：对 logits 做温度缩放，控制采样的随机性。
            logits /= args.task.temperature
            # all_logprobs：对 logits 做 softmax 后取 log，得到每个 token 的 log 概率分布。
            # probs = softmax(logits / temperature)
            # temperature < 1：分布更尖锐，概率更集中，模型更“确定”，生成更保守。
            # temperature > 1：分布更平滑，概率更分散，模型更“随机”，生成更多样。
            all_logprobs = F.log_softmax(logits, dim=-1)
            # logprobs：gather 出每个 response token 的 log 概率，shape 为 [batch, response_length]，即 policy 生成当前 response 的 log 概率。
            logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del output, logits, all_logprobs
            torch.cuda.empty_cache()

            # ref_output：reference policy（通常是SFT模型或初始policy快照）的输出。
            # ref_logits：reference policy 的 logits，取 response 部分。
            # ref_all_logprobs：reference policy 的 log 概率分布。
            # ref_logprobs：reference policy 对 response 的 log 概率，shape 同上。
            ref_output, _ = forward(ref_policy, query_responses, tokenizer)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.task.temperature
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprobs
            torch.cuda.empty_cache()

            # 总结：
            # logprobs 和 ref_logprobs 分别是当前 policy 和 reference policy 对 response 的逐 token log 概率。
            # values 是 policy 对 response 每个 token 的 value 估计。
            # 这些变量是 PPO/RLHF 训练的核心，后续会用于 KL penalty、reward 计算、优势估计和 PPO 损失。
            # **Response Processing**
            # 1. truncate at the first occurrence of `truncate_token` that appears at or after
            # position truncate_after in the responses
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
            # responses：是去掉 prompt（query）之后，模型生成的原始回复（response）token 序列
            truncate_token_mask = responses == args.task.truncate_token
            truncate_after_or_token_mask = torch.cat(
                [
                    torch.zeros_like(truncate_token_mask)[:, : args.task.truncate_after],
                    truncate_token_mask[:, args.task.truncate_after :],
                ],
                dim=1,
            )
            truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
            # postprocessed_responses：是在 responses 基础上，根据任务需求（如遇到特殊截断 token、长度限制等）进一步处理后的回复。
            postprocessed_responses = torch.where(
                truncate_mask,
                torch.full_like(responses, tokenizer.pad_token_id),
                responses,
            )
            del truncate_token_mask, truncate_after_or_token_mask, truncate_mask
            torch.cuda.empty_cache()

            # 2. run reward model on the truncated responses
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            postprocessed_query_responses = right_padding_to_left_padding(
                postprocessed_query_responses, tokenizer.pad_token_id
            )
            scores = get_reward(reward_model, postprocessed_query_responses, tokenizer)[1].flatten()

            # 3. filter response. Ensure that the sample contains truncate_token
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            matches_token = postprocessed_responses[:, args.task.truncate_after :] == args.task.truncate_token
            filter_mask = torch.any(matches_token, dim=-1)
            scores = torch.where(
                filter_mask,
                scores,
                torch.full_like(scores, args.task.penalty_reward_value),
            )
            del matches_token, filter_mask
            torch.cuda.empty_cache()

            # 4. compute rewards
            # 这里的 logprobs 和 ref_logprobs 都是对同一个 response 序列，分别在当前 policy 和 reference policy 下的 log 概率（逐 token）
            # 实际上，这里计算的是逐 token 的对数概率差，它和 KL 散度的关系如下：
            # 1. 逐 token 的 KL penalty
            # 对于每个 token，policy 的概率为 $p$，reference 的概率为 $q$，则 log 概率分别为 $\log p$ 和 $\log q$。
            # $kl = \log p - \log q = \log \frac{p}{q}$
            # 在 PPO/RLHF 训练中，常用的 KL penalty 形式是，用当前 policy 采样出的 response，计算它在 policy 和 reference 下的 log 概率差，
            # 再对 batch 求平均，就是经验 KL。
            # 严格的 KL 散度需要对所有可能的 $x$ 求和（或积分），但在 RLHF/PPO 训练中，我们只关心当前采样到的序列，所以用 log 概率差近似经验 KL。
            # 这种做法在 RLHF 工程和论文中非常常见，既高效又足够有效。用经典 KL 公式，需要用 policy 的概率 $p_i$ 作为权重，乘以 log 概率比，再求和
            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            rewards[:, -1] += scores

            # 5. whiten rewards，把 rewards（或 advantage）标准化为均值 0、方差 1，提升训练稳定性
            # rewards 的 shape 是 [batch_size, response_length]
            # 不是对每个样本单独做标准化，而是对整个 batch 的所有 reward 一起做
            # whiten 是对整个 rewards 矩阵的所有元素整体做标准化，即把所有 64×24 个数拉成一维，计算全局均值和方差，然后标准化
            if args.ppo.whiten_rewards:
                rewards = whiten(rewards, shift_mean=False)

            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                try:
                    all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                    all_postprocessed_query_responses = tokenizer.batch_decode(
                        postprocessed_query_responses, skip_special_tokens=True
                    )
                    all_postprocessed_responses = [
                        x[len(y) :] for x, y in zip(all_postprocessed_query_responses, all_decode_queries)
                    ]

                    kl_sum = kl.sum(axis=1)
                    all_df = pd.DataFrame(
                        {
                            "query": all_decode_queries,
                            "response": all_postprocessed_responses,
                            "score": scores.float().cpu().numpy(),
                            "kl": kl_sum.float().cpu().numpy(),
                            "reward": (scores - kl_ctl.value * kl_sum).float().cpu().numpy(),
                        }
                    )
                    if accelerator.is_main_process and args.track:
                        wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=update)
                    print_rich_table(f"Sample Output at Episode {global_step}", all_df[:4], console)
                except Exception as e:
                    print(e)
                del (
                    all_decode_queries,
                    all_postprocessed_query_responses,
                    all_postprocessed_responses,
                    kl_sum,
                    all_df,
                )
            del postprocessed_query_responses
            torch.cuda.empty_cache()

            # 6. compute advantages and returns
            # GAE， Generalized Advantage Estimation
            lastgaelam = 0
            advantages_reversed = [] # 逆序数组，方便循环计算折扣
            gen_length = args.task.response_length
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                # rewards[:, t] 是第t个token的reward，gamma*nextvalues，下一个token的value，乘以gamma折扣，values[:,t]，当前token的value
                # 用当前token的即时奖励加上未来价值的折扣，减去当前价值，衡量比预期好多少
                delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t] 
                lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            '''
            value
            是 value function 的输出，也叫“状态价值”。
            表示模型估计在当前状态下，未来能获得的期望总回报。
            在 PPO 里，通常是 policy 模型 value head 对每个 token 的 value 估计。
            
            advantage
            表示“当前动作/生成的 token 比模型预期的 value 好多少”。
            计算方式是：实际获得的回报（return）减去模型对当前状态的 value 估计。
            直观理解：如果 advantage > 0，说明实际表现比模型预期好，应该鼓励这种行为；反之则抑制。
            
            return
            是实际获得的回报（reward 累加、折扣后的总和），也叫“目标 value”或“target”。
            在 GAE（Generalized Advantage Estimation）中，return = advantage + value。
            
            advantage = return - value
            所以 return = advantage + value
            这样做的目的是：
            用 advantage 训练 policy（策略），鼓励比预期好的行为。
            用 return 训练 value head，让 value head 预测的 value 更接近实际回报。
            
            在这份代码里，policy模型和critic（value）模型是共享同一个 lm_backbone 的，只是最后接了不同的 head（一个是 logits head，一个是 value/scalar head）
            两个 loss（policy loss 和 value loss）都会对共享的 lm_backbone 反向传播梯度。
            也就是说，无论是 policy head 还是 value head 的 loss，最终都会影响 backbone 的参数。
            这是标准的 actor-critic 合一结构（共享主干），也是 Hugging Face RLHF/PPO 工程的主流做法。
            policy loss 让 backbone 更擅长生成高 reward 的 token。
            value loss 让 backbone 更准确地估计每个 token 的未来回报。
            两者梯度会在 backbone 处“合流”，共同优化主干参数
            '''
            returns = advantages + values
            advantages = whiten(advantages)
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        # epoch控制，对同一批采样数据进行多轮（如4次）优化，提高样本利用率
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.ppo.local_batch_size)
            minibatch_idx = 0
            # 将整个batch（如64条数据）分成多个较小的minibatch（如每个16条），便于分布式和大batch训练。
            # 每次循环取出一个minibatch索引，后续再分microbatch做梯度累积。
            for mini_batch_start in range(0, args.ppo.local_batch_size, args.ppo.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.ppo.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                # 将minibatch再切分为更小的microbatch（如每个4条），每个microbatch前向、反向传播一次，累积梯度，最后统一更新参数。
                for micro_batch_start in range(0, args.ppo.local_mini_batch_size, args.ppo.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.ppo.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_return = returns[micro_batch_inds]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        # 下面为PPO多轮优化阶段的关键变量添加详细注释，并举例说明：
                        # mb_return: 当前microbatch的returns（目标回报），shape=[micro_batch, response_length]。
                        #   例如：mb_return[0] = [1.2, 0.8, 0.5, ...]，表示第0条样本每个token的目标回报。
                        mb_return = returns[micro_batch_inds]
                        # mb_advantage: 当前microbatch的advantage（优势），shape=[micro_batch, response_length]。
                        #   例如：mb_advantage[0] = [0.7, 0.3, -0.2, ...]，表示第0条样本每个token的优势。
                        mb_advantage = advantages[micro_batch_inds]
                        # mb_values: 当前microbatch采样时policy value head输出的value，shape=[micro_batch, response_length]。
                        #   例如：mb_values[0] = [0.5, 0.6, 0.7, ...]，表示第0条样本每个token的采样时value。
                        mb_values = values[micro_batch_inds]
                        # mb_responses: 当前microbatch的response token序列，shape=[micro_batch, response_length]。
                        #   例如：mb_responses[0] = [201, 202, 203, ...]，表示第0条样本的回复token id序列。
                        mb_responses = responses[micro_batch_inds]
                        # mb_query_responses: 当前microbatch的query+response拼接序列，shape=[micro_batch, query+response_length]。
                        #   例如：mb_query_responses[0] = [101, 102, 103, 201, 202, 203, ...]，表示第0条样本的完整输入。
                        mb_query_responses = query_responses[micro_batch_inds]
                        # mb_logprobs: 当前microbatch采样时policy对response的log概率，shape=[micro_batch, response_length]。
                        #   例如：mb_logprobs[0] = [-0.3, -1.2, -2.0, ...]，表示第0条样本每个token的log概率。
                        mb_logprobs = logprobs[micro_batch_inds]
                        # output, vpred_temp: 当前参数下policy对microbatch重新前向传播，output为logits等，vpred_temp为value head输出。
                        #   例如：vpred_temp[0] = [[0.4], [0.5], [0.6], ...]，表示第0条样本每个token的当前value估计。
                        output, vpred_temp = forward(policy, mb_query_responses, tokenizer)
                        # logits: 当前参数下policy对response部分的logits，shape=[micro_batch, response_length, vocab_size]。
                        #   例如：logits[0, 0, :] 是第0条样本第1个token的所有词表分数。
                        logits = output.logits[:, context_length - 1 : -1]
                        # new_all_logprobs: 当前参数下policy对response部分的log概率分布，shape=[micro_batch, response_length, vocab_size]。
                        #   例如：new_all_logprobs[0, 0, :] 是第0条样本第1个token的log概率分布。
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        # new_logprobs: 当前参数下policy对response的log概率，shape=[micro_batch, response_length]。
                        #   例如：new_logprobs[0] = [-0.4, -1.1, -2.2, ...]，表示第0条样本每个token的当前log概率。
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        # vpred: 当前参数下policy value head输出的value，shape=[micro_batch, response_length]。
                        #   例如：vpred[0] = [0.4, 0.5, 0.6, ...]，表示第0条样本每个token的当前value估计。
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        # vpredclipped: 对vpred做clip，防止value更新过大，shape=[micro_batch, response_length]。
                        #   例如：vpredclipped[0] = [0.45, 0.55, 0.65, ...]，表示clip后的value。
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        # vf_losses1: value loss的原始项，shape=[micro_batch, response_length]。
                        #   例如：vf_losses1[0] = [(0.4-1.2)^2, (0.5-0.8)^2, ...]，即(vpred-mb_return)^2。
                        vf_losses1 = torch.square(vpred - mb_return)
                        # vf_losses2: value loss的clip项，shape=[micro_batch, response_length]。
                        #   例如：vf_losses2[0] = [(0.45-1.2)^2, (0.55-0.8)^2, ...]，即(vpredclipped-mb_return)^2。
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        # vf_loss: value loss，取clip前后最大值，防止value更新过大，取均值。
                        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                        # vf_clipfrac: value loss中clip比例，shape=标量。
                        #   例如：vf_clipfrac=0.2，表示有20%的token value loss被clip。
                        vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()
                        # logprobs_diff: 当前log概率与采样时log概率的差，shape=[micro_batch, response_length]。
                        #   例如：logprobs_diff[0] = [0.1, 0.2, -0.1, ...]。
                        logprobs_diff = new_logprobs - mb_logprobs
                        # ratio: 当前概率与采样时概率的比值，shape=[micro_batch, response_length]。
                        #   例如：ratio[0] = [1.1, 1.2, 0.9, ...]。
                        ratio = torch.exp(logprobs_diff)
                        # pg_losses: PPO策略损失原始项，shape=[micro_batch, response_length]。
                        #   例如：pg_losses[0] = [-0.7, -0.3, 0.2, ...]。
                        pg_losses = -mb_advantage * ratio
                        # pg_losses2: PPO策略损失clip项，shape=[micro_batch, response_length]。
                        #   例如：pg_losses2[0] = [-0.7, -0.3, 0.2, ...]，clip后。
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        # pg_loss: PPO最终策略损失，取clip前后最大值，取均值。
                        pg_loss = torch.max(pg_losses, pg_losses2).mean()
                        # pg_clipfrac: 策略损失clip比例，shape=标量。
                        #   例如：pg_clipfrac=0.15，表示有15%的token策略损失被clip。
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        # loss: PPO总损失，policy loss + value loss。
                        loss = pg_loss + args.ppo.vf_coef * vf_loss
                        # 反向传播和优化步骤
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        # prob_dist: 当前logits的softmax概率分布，shape=[micro_batch, response_length, vocab_size]。
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        # entropy: 当前概率分布的信息熵，shape=[micro_batch, response_length]。
                        #   例如：entropy[0] = [2.1, 2.0, 1.8, ...]。
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        # approxkl: 经验KL散度近似，shape=标量。
                        #   例如：approxkl=0.02。
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        with torch.no_grad():
                            approxkls_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            clipfracs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropies_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if accelerator.is_main_process:
                    console.print(
                        f"ppo_epoch_idx",
                        ppo_epoch_idx,
                        "approxkl",
                        approxkl.item(),
                        "pg_loss",
                        pg_loss.item(),
                        "pg_clipfrac",
                        pg_clipfrac.item(),
                        "ratio",
                        ratio.mean().item(),
                    )

        with torch.no_grad():
            if not args.deepspeed:  # for some reason there is a OOM with the `writer.add_histogram`
                writer.add_histogram("ppo/val/ratio_hist", ratio, update)
            kl = logprobs - ref_logprobs
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/value", accelerator.gather(vf_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkls_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(clipfracs_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropies_stats).mean().item(), update)
            writer.add_scalar("ppo/returns/mean", accelerator.gather(return_mean).mean().item(), update)
            writer.add_scalar("ppo/returns/var", accelerator.gather(return_var).mean().item(), update)
            writer.add_scalar("ppo/val/vpred", accelerator.gather(vpred.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/error", accelerator.gather(vf_losses1.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac", accelerator.gather(vf_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/val/mean", accelerator.gather(value_mean).mean().item(), update)
            writer.add_scalar("ppo/val/var", accelerator.gather(value_var).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio.mean()).var().item(), update)
            writer.add_scalar("ppo/val/advantage", accelerator.gather(advantages.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/advantage_var", accelerator.gather(advantages.mean()).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", lrnow, update)
            writer.add_scalar("ppo/episode", global_step, update)
            kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores

    # 保存最终训练好的policy模型到指定路径，支持分布式环境下的安全保存
    accelerator.save_model(policy, args.save_path)

    # 如果是主进程且需要上传模型到HuggingFace Hub
    if accelerator.is_main_process and args.upload_model:
        from huggingface_hub import add_collection_item, create_collection, whoami  # 导入HuggingFace Hub相关API

        # 构建模型仓库名称，包含实验名、数据集、随机种子等信息，便于区分
        repo_name = f"{args.exp_name}__{args.rewards.label_dataset.replace('/', '_')}__seed{args.seed}"
        if not args.hf_entity:
            args.hf_entity = whoami()["name"]  # 自动获取当前用户/组织名
        repo_id = f"{args.hf_entity}/{repo_name}"  # 完整的repo路径
        # 保存主模型权重到HuggingFace Hub，支持安全序列化和推送
        accelerator.unwrap_model(policy).lm_backbone.save_pretrained(
            repo_id, repo_id=repo_id, safe_serialization=True, push_to_hub=True
        )
        # 保存分词器到Hub，保证推理时一致性
        tokenizer.save_pretrained(repo_id, repo_id=repo_id, push_to_hub=True)
        # 创建/获取一个模型集合（collection），便于团队管理多个相关模型
        collection = create_collection(title=f"lm-human-preference-details", namespace=args.hf_entity, exists_ok=True)
        # 将本次训练的模型添加到集合中，方便归档和展示
        add_collection_item(collection.slug, repo_id, item_type="model")


# 程序入口，命令行解析参数并启动训练流程
if __name__ == "__main__":
    args = tyro.cli(Args)  # 使用tyro自动解析命令行参数为Args对象
    train(args)  # 启动主训练流程
