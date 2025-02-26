import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Model, GPT2Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_steps, total_steps, decrease_mode='cosin'):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if decrease_mode == 'cosin':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            elif decrease_mode == 'linear':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 1.0 - progress
            elif decrease_mode == 'const':
                return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def cellular_automaton(rule, width=100, init='random', steps=100, k=1):
    """
    Simulates an elementary cellular automaton.

    Parameters:
    rule (int): The rule number (0-255).
    width (int): The width of the domain (number of cells). Default is 100.
    init (str or list): Initialization method ('random', 'zeros', 'ones', or a list). Default is 'random'.
    steps (int): Number of time steps to simulate. Default is 100.
    k (int): Interval for outputting time points. Default is 1 (every time point).

    Returns:
    list: A list of states at specified time intervals.
    """
    rule_bin = np.array([int(x) for x in np.binary_repr(rule, width=8)], dtype=np.uint8)
    if init == 'random':
        state = np.random.randint(2, size=width)
    elif init == 'zeros':
        state = np.zeros(width, dtype=np.uint8)
    elif init == 'ones':
        state = np.ones(width, dtype=np.uint8)
    elif isinstance(init, list) and len(init) == width:
        state = np.array(init, dtype=np.uint8)
    else:
        raise ValueError("Invalid initialization method")

    states = [state.copy()]
    for _ in range(steps):
        new_state = np.zeros(width, dtype=np.uint8)
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            neighborhood = (left << 2) | (center << 1) | right
            new_state[i] = rule_bin[7 - neighborhood]
        state = new_state.copy()
        if (_ + 1) % k == 0:
            states.append(state.copy())

    return states


def create_sequences_for_pretrain(states, seq_length, k):
    """
    Args:
        states: list of length 'steps'
        seq_length: window of timepoints to select for 1 LLM input sample
        k: how many timepoints in future to skip
    """
    sequences = []
    targets = []
    for i in range(0, len(states) - seq_length * k, k):
        seq = states[i:i + seq_length * k:k]
        target = states[i + k:i + seq_length * k + k:k]  # shifted up 1 sequence element
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)  # [num_dataset_samples, 60, automata_width]


def create_attention_mask(seq_length):
    """
    Create a lower triangular attention mask.
    Args:
        seq_length: int
    """
    mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)
    return mask


class CustomGPT2Model(nn.Module):
    def __init__(self, input_size, n_embd, n_layer, n_head, seq_length):
        super(CustomGPT2Model, self).__init__()
        self.config = GPT2Config(
            n_embd=n_embd,
            vocab_size=1,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=seq_length,
        )
        _gpt2 = GPT2Model(self.config)
        self.wpe = _gpt2.wpe
        self.gpt2 = _gpt2.h
        self.dropout = _gpt2.drop
        self.ln_f = _gpt2.ln_f

        self.seq_length = seq_length
        self.input_projection = nn.Linear(input_size, n_embd)
        self.output_layer = nn.Linear(n_embd, input_size)

    def forward(self, input_sequences, attention_mask, output_attentions=False):
        """
        Parameters:
        input_sequences: (b, l, [100])

        Returns:
        logits: (b, l, [100]), no sigmoid activation
        """
        b, l, _ = input_sequences.shape
        input_embeds = self.input_projection(input_sequences)
        hidden_states = self.dropout(self.wpe(torch.arange(l).to(input_sequences.device)) + input_embeds)
        attentions = []
        for i in range(self.config.n_layer):
            out = self.gpt2[i](hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = out[0]
            attentions.append(out[1])
        hidden_states = self.ln_f(hidden_states)

        logits = self.output_layer(hidden_states)
        if output_attentions:
            return logits, attentions
        return logits


@dataclass
class CustomGPT2Config:
    input_size: int
    seq_length: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


gpt2_config_map = {
    'tiny': dict(n_layer=1, n_head=1, n_embd=64),  # 857,188
    'small': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params, 124,439,808 - 38597376 | 85,256,548
    'large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params;                      | 708,724,580
}


def create_gpt2_model(gpt2name, input_size=100, seq_length=60):
    config = CustomGPT2Config(input_size, seq_length, **gpt2_config_map[gpt2name])
    return CustomGPT2Model(
        config.input_size,
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.seq_length,
    )


class RGBModel(nn.Module):
    def __init__(self, input_size, gpt2: CustomGPT2Model, num_classes=4):
        """
        :param input_size: sample width
        :param gpt2:  CustomGPT2Model
        :param num_classes:
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_proj = nn.Linear(input_size, gpt2.input_projection.in_features)
        self.output = nn.Linear(gpt2.output_layer.out_features, gpt2.output_layer.out_features * num_classes)
        self.gpt2 = gpt2
        self.freeze()

    def forward(self, x, attention_mask):
        b, l, _ = x.size()
        x = self.input_proj(x)
        x = self.gpt2(x, attention_mask)
        return self.output(x).reshape(b, l, -1, self.num_classes)

    def freeze(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    for size in ['small', 'medium', 'large']:
        model = create_gpt2_model(size)
        print(model)
        print(model.config)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print('=' * 100)
