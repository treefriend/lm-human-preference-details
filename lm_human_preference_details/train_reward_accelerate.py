import functools
import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs, broadcast
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import Tensor, optim
from torch.optim.optimizer import (
    _dispatch_sqrt,
    _get_value,
    _use_grad_for_differentiable,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


'''
# @dataclass 是 Python 3.7+ 标准库 dataclasses 提供的一个装饰器，用于简化类的定义，让你可以更方便地创建只用于存储数据的类（数据类）
# 加上 @dataclass 后，Python 会自动为这个类生成常用方法，比如 __init__（构造函数）、__repr__、__eq__ 等，无需手动写这些模板代码。
原来这代码要这样写：
class LabelHParams:
    def __init__(self, type=None, num_train=4992, num_labels=4, source=None):
        self.type = type
        self.num_train = num_train
        self.num_labels = num_labels
        self.source = source
不过现在更流行用pydantic，因为还会做一些类型检查啥的，一样简洁，更安全
from pydantic import BaseModel, Field

class LabelHParams(BaseModel):
    type: str = None
    num_train: int = 4992
    num_labels: int = 4
    source: str = None
'''
@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 4992
    num_labels: int = 4
    source: str = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 64
    query_dataset: str = "books"

    # Response params
    response_length: int = 24

    # LM params
    temperature: float = 0.7


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanrl"
    """the wandb's project name
    （Weights & Biases）是一个机器学习实验管理和可视化工具，常用于深度学习训练过程中的：
    日志记录：自动记录训练过程中的 loss、accuracy、学习率等指标。
    可视化：提供交互式网页界面，实时可视化训练曲线、超参数、模型结构等。
    实验对比：方便对比不同实验的结果，追踪超参数和模型表现。
    团队协作：支持团队成员共享实验结果和模型。
    在你的代码里，wandb 用于跟踪和可视化 reward model 训练过程中的各种指标，帮助你更好地分析和调优模型。
    """
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model
    是微软开源的一个深度学习训练优化库，主要用于加速和扩展大规模分布式训练，尤其适合大模型（如 GPT、BERT 等）的高效训练
    支持数据并行、模型并行、流水线并行等多种并行方式，能让模型在多卡、多机甚至超大规模集群上高效训练。
    通过 ZeRO 技术大幅降低单卡显存占用，使得单张显卡也能训练超大模型。
    集成了混合精度训练（FP16/BF16）、梯度累积、异步数据加载等优化手段，提升训练速度。
    """
    label_dataset: str = "sentiment/offline_5k.json"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    local_batch_size: int = 4
    """per rank batch size"""
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    lr: float = 0.00005
    """the learning rate"""
    eps: float = 1e-5
    """the epsilon for Adam"""
    local_rollout_batch_size: int = 512
    """per rank rollout batch size"""
    rollout_batch_size: tyro.conf.Suppress[int] = None
    """rollout batch size"""
    world_size: tyro.conf.Suppress[int] = None
    """the number of processes to use"""
    batch_size: tyro.conf.Suppress[int] = None
    """the batch size across all ranks"""
    local_normalize_samples: int = 256
    """Samples used to estimate reward mean and std"""
    normalize_samples: tyro.conf.Suppress[int] = None
    """Samples used to estimate reward mean and std across all ranks"""
    debug_normalize: int = 0
    """Samples used to check that normalization worked"""
    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/reward"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


OPENAI_PAD_TOKEN_ID = 50259


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
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
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        step = _get_value(step_t)

        ### pytorch adam implementation:
        # bias_correction1 = 1 - beta1 ** step
        # bias_correction2 = 1 - beta2 ** step
        # step_size = lr / bias_correction1
        # bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        # param.addcdiv_(exp_avg, denom, value=-step_size)

        ### tensorflow adam implementation:
        lr_t = lr * _dispatch_sqrt(1 - beta2**step) / (1 - beta1**step)
        denom = exp_avg_sq.sqrt().add_(eps)
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
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        self.lm_backbone = lm_backbone
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward_latents = output.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]
        '''
        在语言模型中，reward_latents 通常是每个 token 的隐藏状态。-1 代表最后一个 token 的隐藏状态，也就是模型对整个输入序列“理解”后的输出。
        BERT 这类双向 Transformer 通常用第一个 token（CLS token）的隐藏状态作为整个句子的 embedding，因为 BERT 在预训练和下游任务时就是这样设计和训练的。
        GPT、GPT-2、GPT-3 这类自回归语言模型（即 Causal LM）和 BERT 不同，它们没有专门的 CLS token，通常用最后一个 token 的隐藏状态来代表整个序列的“总结”或输出。
        在 reward model 或生成式任务中，最后一个 token 的隐藏状态包含了模型对整个输入序列的全部上下文理解，适合用来做 reward/value 预测。
        这种做法在 RLHF、reward modeling、生成式模型等场景很常见。一般是<EOS>，有时候基于数据处理也可能是<PAD>或者其他token
        如果是packing方式训练，可以在每个句子pack的地方用特殊分隔符<SEP>，可以把它的embedding拿出来，每个句子都加<EOS>也可以
        '''
        last_reward_latents = reward_latents[:, -1, :]
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        # shape: [batch_size, 1]
        '''
        它们虽然是可训练参数，但 reward model 的 loss（如交叉熵）对 gain/bias 的梯度很小，实际训练时 gain/bias 基本不会有显著变化。
        在训练前后，代码会调用 normalize() 函数，手动根据 reward 的均值和标准差重新赋值
        归一化操作会直接覆盖掉训练过程中 gain/bias 的值，确保 reward 分布始终符合目标（如均值 0，方差 1）。
        只要每次训练前后都做一次归一化，reward_gain 和 reward_bias 就不会“跑偏”。
        '''
        reward = self.reward_gain * reward + self.reward_bias
        return output, reward


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding.
    不同模型和训练代码对 padding 方式有不同要求。有的模型（如 GPT-2）习惯左补齐（左边填充 PAD），有的习惯右补齐（右边填充 PAD）
    Hugging Face 的 tokenizer 默认是右补齐（padding_side="right"），但有些生成模型或 reward model 只支持左补齐输入。
    原始右补齐：
    [1, 2, 3, 0, 0]（假设 0 是 pad_id）
    转换后左补齐：
    [0, 0, 1, 2, 3]
    GPT 这类自回归模型通常假设输入的有效 token 都在序列的右侧，左边是 padding。这样，模型的 position_ids 从左到右递增，padding 部分不会影响有效 token 的预测。
    如果用右补齐，padding 会出现在序列末尾，模型在生成时可能会把 padding 当作有效输入，导致位置编码和 attention mask 混乱。
    左补齐时，有效 token 的 position_ids 是连续的，padding 的 position_ids 可以统一设为 0 或忽略，方便实现。
    右补齐时，有效 token 的 position_ids 不是从 0 开始，处理起来更麻烦。
    左补齐可以让所有样本的最后一个 token 对齐，方便 batch 处理和并行计算，尤其适合自回归生成任务。
    Hugging Face GPT-2/3 等模型默认只支持左补齐
    """
    assert tokens.ndim == 2
    return torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    )

# ceil_div 用于确保即使最后一批数据不满，也能分配到一个 batch，常见于数据分割、批量训练等场景。常用于分批处理时，确定需要多少个 batch 能覆盖全部数据。
# -1的目的是确保a整除b的时候不出意外
def ceil_div(a, b):
    return (a - 1) // b + 1

# 确保a可以整除b，不行就报错
# 在分 batch、分组、梯度累积等场景下，代码有时要求数据量必须能被 batch size 或 step size 整除，否则后续逻辑会出错。
# 用 exact_div 可以在程序运行时提前发现这种错误，避免隐蔽 bug。
def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens
    用语言模型（lm_backbone）对输入 queries 进行生成，并保证 padding token 不被模型生成和污染。
    后续代码中，手动添加了pad token，这在gpt2中本来么有的
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)是把 padding 的位置都替换成 0。
    这样做的目的是确保 padding 位置的 token id 是一个确定的值，不会影响模型生成和 loss 计算。
    但如果你的 pad_token_id 不是 0，这里用 0 其实并不严谨，最好用 tokenizer.pad_token_id 替换，这样更通用
    """
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
    '''
    cumsum(1) - attention_mask.long() 就是不包含当前 token 的有效 token 个数，即“exclusive cumsum”（不包括当前位置本身）。
    [0, 0, 101, 102, 103]   # 0是pad，101/102/103是有效token
    attention_mask: [0, 0, 1, 1, 1]
    cumsum(1):      [0, 0, 1, 2, 3]
    exclusive:      [0, 0, 0, 1, 2]
    这正好符合 GPT-2/3 等自回归模型对 position_ids 的要求。
    '''
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
循环多次，从数据集中采样 queries，调用 generate 生成模型回复（推理），得到若干组 query-response
对 reward model 的输出进行归一化（normalize），确保 reward 分布的均值为 0、标准差为 1
只有在真实数据上多次推理，才能获得准确的 reward 分布统计，保证归一化后 reward 的尺度和分布稳定，便于后续 RLHF/PPO 等算法训练。
这种做法类似于“采样校准”，是 RLHF reward modeling 的常见实践。
reward 模型在训练过程中参数会不断更新，那么归一化用的均值和标准差怎么确保和最新的数据分布一致？
函数会在训练前和训练后都调用一次
训练前后都归一化是最常见做法。
如果 reward model 后续还会继续训练或微调，建议每次训练后都重新归一化一次。
reward（奖励）归一化是非常重要的
1. 保证奖励分布稳定，便于下游训练
    不同 reward model 输出的奖励分布（均值、方差）可能差异很大。
    如果 reward 分布尺度不统一，RL/PPO 等算法的训练会变得不稳定，甚至无法收敛。
    归一化后 reward 的均值为 0、标准差为 1，可以让训练过程更平滑，超参数（如 KL 系数）更容易迁移和调优。
2. 便于不同实验、模型之间的对比
    归一化 reward 后，不同实验、不同模型的奖励输出有了统一的“量纲”，结果才有可比性。
    这样可以公平地比较不同 reward model 或 RL 策略的效果。
3. 防止 reward “漂移”或“爆炸”
    如果 reward 分布偏移（均值很大或很小），RL 算法可能会学到无意义的策略，甚至训练发散。
    归一化可以防止 reward 爆炸或塌陷，提升训练的鲁棒性。
4. 理论和工程实践的共识
    归一化 reward 是 RLHF、PPO、reward modeling 等领域的标准做法，几乎所有主流实现都会这样处理。
    这也是 OpenAI、Anthropic 等大模型 RLHF pipeline 的通用工程规范。
'''
def normalize(
    args,
    accelerator,
    device,
    lm_backbone,
    reward_model,
    iter_dataloader,
    generation_config,
    tokenizer,
):
    with torch.no_grad():
        # reset reward scales
        accelerator.unwrap_model(reward_model).reward_gain.data.fill_(1.0)
        accelerator.unwrap_model(reward_model).reward_bias.data.fill_(0.0)
        # number of minibatches for computing the normalization statistics
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        # 读代码可以祛魅，可以战胜技术焦虑啊
        # 先把采样数据的batch手动拼起来
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)

        # compute reward statistics
        # 得到reward的值
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        # 全局的均值和标准差
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")

        # reward normalization
        # 正则化，N(0,1)
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        gain = target_std / std
        bias = target_mean - gain * mean
        print(f"gain: {gain}, bias: {bias}")
        accelerator.unwrap_model(reward_model).reward_gain.data = gain
        accelerator.unwrap_model(reward_model).reward_bias.data = bias

        # validate normalization
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")


def train(args: Args):
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=False,
            )
        ],  # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.local_micro_batch_size = exact_div(args.local_batch_size, args.gradient_accumulation_steps)

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            wandb.run.log_code(".")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    '''
    1. 为什么不用训练数据直接归一化 reward？
    直接用训练数据（比如人类偏好标注的 query-response 对）喂给 reward model，理论上可以，但这样做有一些局限：
        训练数据的分布和后续 RLHF/PPO 训练时 policy model 生成的数据分布可能不一致。
        训练数据通常是“人工标注的好样本”，而 RLHF/PPO 训练时，policy model 会生成各种“好坏混杂”的 response，分布更广。
        如果只用训练数据归一化，reward model 在实际 RLHF/PPO 训练时遇到的 reward 分布可能会“漂移”，导致归一化失效，影响训练稳定性。
    2. 为什么要用 untrained model 生成 response 归一化？
        untrained model（通常是 base model，如 GPT-2）生成的 response，更接近 RLHF/PPO 训练初期 policy model 生成的分布。
        用 untrained model 采样，可以让 reward model 的归一化统计量（均值、方差）更贴近实际 RLHF/PPO 训练时的输入分布，
        保证归一化后的 reward 在 RLHF/PPO 训练中依然是标准正态分布。
        这样做可以提升 RLHF/PPO 训练的稳定性和泛化性，也是 OpenAI、Anthropic 等主流 RLHF pipeline 的工程规范。
    3. 完全可以直接使用后续进行 PPO 的 policy model 作为 untrained model 来做 reward 归一化采样，而且这在实际 RLHF 工程中是很常见、合理的做法。
        PPO 阶段的 policy model（即 SFT 预训练好的模型）生成的 response 分布，和你后续 RLHF/PPO 训练时遇到的数据分布是一致的。
        用它来采样归一化 reward，可以让 reward 的均值和方差更贴合实际 RLHF/PPO 训练时的 reward 分布，归一化效果更科学、更稳定。
    '''
    untrained_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    untrained_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    untrained_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    reward_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reward_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # make sure the `lm_head` or `embed_out` does not require gradients, otherwise
    # pytorch DDP complains; see https://gist.github.com/vwxyzjn/45fc8706dfb3cf33695f0f57cc44a533
    if isinstance(reward_model.lm_backbone, transformers.GPTNeoXForCausalLM):
        reward_model.lm_backbone.embed_out.requires_grad_(False)
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(reward_model.parameters(), lr=args.lr, eps=args.eps)
    else:
        optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    dataset = load_dataset("bookcorpus", split="train")
    dataset = dataset.shuffle(seed=local_seed)

    def process_query_data(x, base_model: str, response_length: int):  # added args so it's hashable
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return {
            "query_token": tokenizer(
                x["text"], padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
            )["input_ids"],
        }

    dataset.set_transform(
        functools.partial(process_query_data, base_model=args.base_model, response_length=args.task.response_length)
    )
    dataloader = DataLoader(dataset, batch_size=args.local_rollout_batch_size)
    reward_model, optimizer, dataloader = accelerator.prepare(reward_model, optimizer, dataloader)
    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.ppo.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            # "steps_per_print": 10,
            # "zero_optimization": {
            #     "stage": stage,
            #     "stage3_param_persistence_threshold": 1e4,
            #     "offload_param": {
            #         "device": off_load_device
            #     }
            # },
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        untrained_model, *_ = deepspeed.initialize(model=untrained_model, config=eval_ds_config)
        untrained_model.eval()
    else:
        untrained_model = untrained_model.to(device)

    def repeat_generator():  # TODO: ideally we shuffle the dataloader as well
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    if args.normalize_before:
        print("===Normalize reward model *before* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            args,
            accelerator,
            device,
            untrained_model.lm_backbone,
            reward_model,
            iter_dataloader,
            generation_config,
            tokenizer,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label = load_dataset(
        "vwxyzjn/lm-human-preferences",
        data_files=[args.label_dataset],
    )["train"]
    print("Num labels found in source:", len(label))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    print("===training reward model===")
    all_inds = np.random.permutation(args.labels.num_train)
    # ensure that all processes have the same shuffled indices
    all_inds = broadcast(torch.tensor(all_inds, device=device), 0)
    all_inds = all_inds.cpu().numpy()
    global_step = 0
    for start in range(0, args.labels.num_train, args.batch_size):
        # linear rate annealing
        lr = (1 - start / args.labels.num_train) * args.lr
        optimizer.param_groups[0]["lr"] = lr

        global_step += 1
        end = start + args.batch_size
        b_inds_all = all_inds[start:end]
        b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
        losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
        accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
        gradient_accumulation_step = 0
        # 主训练循环：遍历训练数据，分 batch、分 micro batch 训练 reward_model。
        for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
            '''
            为什么多分类任务不使用bert这种encoder based model？
            实际 RLHF/PPO 阶段，reward model 需要对 policy model 生成的完整 query-response 序列打分。
            这些序列通常是自回归生成的，长度和结构不固定，和 GPT 这类 decoder-only 模型的输入形式天然兼容。
            RLHF pipeline 通常直接用和 policy model（如 GPT-2/3/NeoX）结构一致的 decoder-only reward model，
            这样可以直接利用预训练权重、tokenizer、position embedding 等，且推理流程一致。
            Hugging Face 等主流 RLHF 工程实现，reward model 通常就是在 GPT-2/3/NeoX 这类 decoder-only backbone 上加一个 reward head。
            这样可以直接复用 tokenizer、数据处理、生成流程，且 reward model 可以方便地和 policy model 共享参数或做对比。
            在 RLHF/PPO 等生成式任务中，reward model 通常需要和 policy model 输入形式一致，decoder-only（如 GPT）更主流、更兼容实际 pipeline。
            工程上也更方便直接用 decoder-only backbone。
            '''
            with accelerator.accumulate(reward_model):
                micro_batch_end = micro_batch_start + args.local_micro_batch_size
                micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                mb_data = label[micro_batch_inds]
                mb_query = torch.from_numpy(np.stack(mb_data["query"])).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                mb_responses = [
                    torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)
                ]
                # hack: deal with openai's padding token
                mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                for item in mb_responses:
                    item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

                predicted_rewards = []
                # 在训练 reward model 的时候，每个样本的多个候选 response 会分别和同一个 query 拼接起来，
                # 然后分别送入 reward model，让 reward model 对每个 query-response 组合分别打一个分数（reward）
                for i in range(args.labels.num_labels):
                    query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    query_responses = right_padding_to_left_padding(query_responses, tokenizer.pad_token_id)
                    reward = get_reward(reward_model, query_responses, tokenizer)[1]
                    predicted_rewards.append(reward.view(-1))
                predicted_rewards = torch.stack(
                    predicted_rewards, dim=1
                )  # shape (batch_size, num_labels), basically a reward prediction for each label
                # 每一行是一个query下所有候选response的reward分数
                accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                # loss 计算：reward_model 对每个 query-response 组合打分，使用交叉熵损失训练 reward_model 让其学会区分“好”与“坏”。
                # 交叉熵 loss 反映了模型把概率分配给正确标签的能力。但 softmax 是对所有候选 response 做的，loss 只用正确标签的概率。
                loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
                accelerator.backward(loss)
                optimizer.step()  # accelerate handles gradient accumulation automatically
                optimizer.zero_grad()
                losses[gradient_accumulation_step] = loss
                accuracies[gradient_accumulation_step] = accuracy
            gradient_accumulation_step += 1

        writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step)
        writer.add_scalar("train/accuracy", accelerator.gather(accuracies).mean().item(), global_step)
        writer.add_scalar("train/lr", lr, global_step)
        # 定期评估reward model的准确率，生成样本并计算KL散度然后进行可视化输出
        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            with torch.no_grad():
                # eval on test_label, some duplicate code (I don't want to make the training loop into a function...)
                test_accuracies = []
                len_labels = (len(label) // args.batch_size) * args.batch_size  # in case the last batch is not full
                new_all_inds = np.arange(len_labels)
                for start in range(args.labels.num_train, len_labels, args.batch_size):
                    end = start + args.batch_size
                    b_inds_all = new_all_inds[start:end]
                    b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
                    for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                        mb_data = label[micro_batch_inds]
                        mb_query = torch.from_numpy(np.stack(mb_data["query"]))
                        mb_query = right_padding_to_left_padding(mb_query, tokenizer.pad_token_id).to(device)
                        mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                        mb_responses = [
                            torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)
                        ]
                        # hack: deal with openai's padding token
                        mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                        for item in mb_responses:
                            item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                        predicted_rewards = []
                        for i in range(args.labels.num_labels):
                            query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                            query_responses = right_padding_to_left_padding(query_responses, tokenizer.pad_token_id)
                            reward = get_reward(reward_model, query_responses, tokenizer)[1]
                            predicted_rewards.append(reward.view(-1))
                        predicted_rewards = torch.stack(
                            predicted_rewards, dim=1
                        )  # shape (batch_size, num_labels), basically a reward prediction for each label
                        accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                        test_accuracies.append(accuracy)
                test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
                writer.add_scalar("test/accuracy", test_accuracy, global_step)
                if accelerator.is_main_process:
                    print("test/accuracy", test_accuracy, global_step)

                # the part below is testing out some generations and KLs, not presented in the original code
                data = next(iter_dataloader)
                queries = data["query_token"].to(device)
                context_length = queries.shape[1]
                queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
                query_responses = generate(
                    accelerator.unwrap_model(reward_model).lm_backbone,
                    queries,
                    tokenizer,
                    generation_config,
                )
                responses = query_responses[:, context_length:]

                output, reward = get_reward(reward_model, query_responses, tokenizer)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()

                output, _ = get_reward(untrained_model, query_responses, tokenizer)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                ref_logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()
                '''
                KL 散度越大，说明 reward model 的输出分布和 base model 差异越大，reward model 学到了更多“偏好”信息,
                即 reward head 已经根据人类偏好数据发生了较大调整。监控 reward model 是否过拟合或训练异常（比如 KL 爆炸或塌陷）。
                untrained model 的 reward head（scalar_head）是随机初始化的，没有经过训练,
                但它的 backbone（如 GPT-2）是预训练的，只是最后 reward head 没有学过人类偏好信号。
                所以 untrained model 的 reward 分布可以看作是“无偏好”的 baseline，reward model 则是“有偏好”的版本。 可以当做熵增来理解。
                计算 KL 是为了衡量 reward model 训练后和 base model（无偏好）的输出分布差异，帮助你监控 reward model 是否学到了有效的区分能力。
                '''
                kl = logprobs - ref_logprobs
                kl_sum = kl.sum(axis=1)
                all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                all_query_responses = tokenizer.batch_decode(query_responses, skip_special_tokens=True)
                all_responses = [x[len(y) :] for x, y in zip(all_query_responses, all_decode_queries)]
                all_df = pd.DataFrame(
                    {
                        "query": all_decode_queries,
                        "response": all_responses,
                        "kl": kl_sum.float().cpu().numpy(),
                    }
                )
                if accelerator.is_main_process and args.track:
                    wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=global_step)
                print_rich_table(f"Sample Output at Step {global_step}", all_df[:4], console)
                del (
                    query_responses,
                    all_decode_queries,
                    all_query_responses,
                    all_responses,
                    kl_sum,
                    all_df,
                )
                writer.add_scalar("train/kl", kl.sum(1).mean().item(), global_step)

    torch.cuda.empty_cache()
    if args.normalize_after:
        print("===Normalize reward model *after* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            args,
            accelerator,
            device,
            untrained_model.lm_backbone,
            reward_model,
            iter_dataloader,
            generation_config,
            tokenizer,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # save model
    if args.save_path:
        accelerator.save_model(reward_model, args.save_path)

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
