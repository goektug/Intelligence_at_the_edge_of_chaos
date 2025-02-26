import os
import argparse
import json
import shutil
import numpy as np
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from utils import (
    set_seed,
    get_lr_scheduler,
    cellular_automaton,
    create_sequences_for_pretrain,
    create_attention_mask,
    create_gpt2_model,
)


class PreTrainDataset(Dataset):
    def __init__(self, sequences, targets):
        super().__init__()
        self.sequences, self.targets = sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]).float(), torch.tensor(self.targets[idx]).float()


def collate_fn(batch, sample_width, width=1000, num_samples=100):
    """ multi version """
    x = []
    y = []

    start_ids = np.random.choice(width - sample_width, num_samples, replace=False).tolist()

    for sample in batch:
        for start in start_ids:
            end = start + sample_width
            x.append(sample[0][:, start:end])
            y.append(sample[1][:, start:end])
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


def build_dataloader(args):
    states = cellular_automaton(args.rule, args.width, args.automation_type, args.steps)
    flattened_states = np.stack(states)
    sequences, targets = create_sequences_for_pretrain(flattened_states, args.seq_length, args.k)

    val_index = int(len(sequences) * 0.9)
    train_sequences, train_targets = sequences[:val_index], targets[:val_index]
    val_sequences, val_targets = sequences[val_index:], targets[val_index:]

    train_dataset = PreTrainDataset(train_sequences, train_targets)
    val_dataset = PreTrainDataset(val_sequences, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, sample_width=args.sample_width,
                                                 num_samples=args.num_samples, width=args.width))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=partial(collate_fn, sample_width=args.sample_width, num_samples=args.num_samples,
                                               width=args.width))

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, scheduler, args):
    model.train()

    train_loss_his = []
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    for batch_sequences, batch_targets in tqdm(train_loader):
        batch_sequences, batch_targets = batch_sequences.to(args.device), batch_targets.to(args.device)
        attention_mask = create_attention_mask(batch_sequences.size(1)).repeat(batch_sequences.size(0), 1, 1, 1).to(
            args.device)

        for sub_batch_sequences, sub_batch_targets, sub_attention_mask in zip(
                torch.chunk(batch_sequences, args.num_samples),
                torch.chunk(batch_targets, args.num_samples),
                torch.chunk(attention_mask, args.num_samples)
        ):
            mb_loss = 0
            for micro_batch_sequences, micro_batch_targets, micro_attention_mask in zip(
                    torch.chunk(sub_batch_sequences, gradient_accumulation_steps),
                    torch.chunk(sub_batch_targets, gradient_accumulation_steps),
                    torch.chunk(sub_attention_mask, gradient_accumulation_steps),
            ):
                logits = model(micro_batch_sequences, micro_attention_mask)
                loss = criterion(torch.sigmoid(logits), micro_batch_targets)
                loss = loss / gradient_accumulation_steps
                mb_loss += loss.item()
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss_his.append(mb_loss)
            wandb_log({"train loss": mb_loss}, args)
        wandb_log({"lr": round(scheduler.get_last_lr()[0], 9)}, args)
        scheduler.step()

    return train_loss_his


def validation_epoch(model, data_loader, criterion, args, mode='val'):
    model.eval()
    total_loss = 0.0
    total_batch = 0
    total_acc, total_auc = 0.0, 0.0
    with torch.no_grad():
        for batch_sequences, batch_targets in tqdm(data_loader):
            batch_sequences, batch_targets = batch_sequences.to(args.device), batch_targets.to(args.device)
            attention_mask = create_attention_mask(batch_sequences.size(1)).repeat(batch_sequences.size(0), 1, 1, 1).to(
                args.device)

            for sub_batch_sequences, sub_batch_targets, sub_attention_mask in zip(
                    torch.chunk(batch_sequences, args.num_samples),
                    torch.chunk(batch_targets, args.num_samples),
                    torch.chunk(attention_mask, args.num_samples)):
                # (b, l, [100])
                logits = model(sub_batch_sequences, sub_attention_mask)
                logits = torch.sigmoid(logits)
                loss = criterion(logits, sub_batch_targets)
                total_loss += loss.item()

                curr_acc, curr_auc = cal_metrics(
                    logits.cpu().numpy(),
                    sub_batch_targets.cpu().numpy()
                )
                total_acc += curr_acc
                total_auc += curr_auc
                total_batch += 1
                if mode == 'train':
                    break

    avg_loss = total_loss / total_batch
    avg_acc = total_acc / total_batch
    avg_auc = total_auc / total_batch
    return avg_loss, avg_acc, avg_auc


def cal_metrics(logits, targets):
    """
    Parameters:
    logits: (b, l, [100])
    targets: (b, l, [100])
    """
    b, l, d = logits.shape
    preds = (logits > 0.5).astype(float)

    # acc = ((preds == targets).sum(-1) == d).sum() / b / l  # by board
    acc = (preds == targets).sum() / b / l / d  # by cell

    try:
        auc = roc_auc_score(targets.reshape(-1), logits.reshape(-1))
    except ValueError:
        auc = -1

    return acc, auc


def wandb_log(data, args):
    if args.wandb_enable:
        wandb.log(data)
    else:
        print(data)


def args_post_init(args):
    if args.wandb_name is None:
        args.wandb_name = f"loss_for_{args.rule}_{args.gpt2_size}"
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.micro_batch_size = args.batch_size if args.micro_batch_size is None else args.micro_batch_size
    args.metrics_path = f"{args.save_dir}/metrics_{args.rule}_{args.k}_{args.gpt2_size}.json"
    args.save_dir = f"{args.save_dir}/rule{args.rule}-k{args.k}-{args.gpt2_size}"
    return args


def save_checkpoint(epoch, count, model, optimizer, scheduler, best_val_loss, best_val_loss_el, save_dir,
                    is_best=False):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'count': count,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_val_loss_el': best_val_loss_el,
    }
    if is_best:
        torch.save(ckpt, os.path.join(save_dir, f'best_pytorch_model_{epoch}_rule{args.rule}-k{args.k}-{args.gpt2_size}.pth'))
    else:
        torch.save(ckpt, os.path.join(save_dir, f'pytorch_model_{epoch}_rule{args.rule}-k{args.k}-{args.gpt2_size}.pth'))


def sort_checkpoint(ck_name):
    if "rule" in ck_name:
        return int(ck_name.replace('.pth', '').split('_')[-2])
    else:
        return int(ck_name.replace('.pth', '').split('_')[-1])


def load_last_checkpoint(model, optimizer, scheduler, save_dir):
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('pytorch_model_')]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found")
    checkpoints.sort(key=lambda x: sort_checkpoint(x))
    sd = torch.load(os.path.join(save_dir, checkpoints[-1]))
    model.load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['optimizer'])
    scheduler.load_state_dict(sd['scheduler'])
    count = sd['count']
    epoch = sd['epoch']
    best_val_loss = sd['best_val_loss']
    best_val_loss_el = sd['best_val_loss_el']
    return model, optimizer, scheduler, count, epoch, best_val_loss, best_val_loss_el


def delete_oldest_checkpoint(save_dir, save_total_limit=2):
    for prefix in ['pytorch_model_', 'best_pytorch_model_']:
        # List all checkpoint dirs
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        # Sort checkpoints by modification time (oldest first)
        checkpoints.sort(key=lambda x: sort_checkpoint(x))
        # If there are more than max_checkpoints, delete the oldest one
        if len(checkpoints) > save_total_limit:
            oldest_checkpoint = checkpoints[0]
            oldest_path = os.path.join(save_dir, oldest_checkpoint)
            print(f'Remove checkpoint: {oldest_path}')
            # shutil.rmtree(oldest_path)
            os.remove(oldest_path)


def main(args):
    args = args_post_init(args)
    set_seed(args.seed)
    if args.wandb_enable:
        if args.resume_run_id is None:
            wandb.init(project=args.wandb_project, name=args.wandb_name,
                    config=args.__dict__)
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=args.__dict__,
                id=args.resume_run_id,
                resume="must"
            )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = create_gpt2_model(args.gpt2_size, args.sample_width, args.seq_length)
    print(model.config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    model.to(args.device)

    _train_loader, _ = build_dataloader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=len(_train_loader) * int(0.1 * args.epochs),
        total_steps=len(_train_loader) * args.epochs,
    )
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_val_loss_el = float('inf')
    if args.patience == -1:
        args.patience = float('inf')
    patience = args.patience
    assert patience == 15
    count = 0
    extra_counter = 0
    extra_training_triggered = False

    start_epoch = 0
    if args.resume:
        model, optimizer, scheduler, count, start_epoch, best_val_loss, best_val_loss_el = load_last_checkpoint(
            model, optimizer, scheduler, args.save_dir
        )

    all_metrics = {
        'train_loss_his': [],
        'train_metric': [],
        'val_metric': [],
        'early_stop_metric': [],
        'patience_metric': [],
    }
    # save local metric
    for epoch in range(start_epoch, args.epochs):
        print(f"==================== epoch: {epoch + 1} / {args.epochs} ===========================")
        train_loader, val_loader = build_dataloader(args)

        train_loss_his = train_epoch(model, train_loader, criterion, optimizer, scheduler, args)
        all_metrics['train_loss_his'].append({epoch: train_loss_his})

        train_loss, train_acc, train_auc = validation_epoch(model, train_loader, criterion, args, mode='train')
        wandb_log({"epoch": epoch + 1, "train acc": train_acc}, args)
        wandb_log({"epoch": epoch + 1, "train auc": train_auc}, args)
        all_metrics['train_metric'].append({epoch: [train_loss, train_acc, train_auc]})

        val_loss, val_acc, val_auc = validation_epoch(model, val_loader, criterion, args)
        wandb_log({"epoch": epoch + 1, "val loss": val_loss}, args)
        wandb_log({"epoch": epoch + 1, "val acc": val_acc}, args)
        wandb_log({"epoch": epoch + 1, "val auc": val_auc}, args)
        all_metrics['val_metric'].append({epoch: [val_loss, val_acc, val_auc]})

        print(f"rule={args.rule}, k={args.k}")
        print(train_loss, train_acc, train_auc)
        print(val_loss, val_acc, val_auc)

        if extra_training_triggered:
            if val_loss >= 1e-3:
                extra_training_triggered = False
                extra_counter = 0
            else:
                extra_counter += 1
                if extra_counter >= 500:
                    print(f"Early stop due to validation loss < 1e-5 and 500 extra epochs at {epoch + 1} epoch!")
                    wandb_log({"epoch": epoch + 1, "Early stop": epoch + 1}, args)
                    all_metrics['early_stop_metric'].append(epoch)
                    break
        if val_loss < 1e-3 and not extra_training_triggered:
            extra_training_triggered = True

        # early stop
        if (epoch + 1) % 100 == 0:
            # val loss increasing or decreasing a few
            if val_loss - best_val_loss_el >= -1e-4:
                count += 1
                if count > patience:
                    print(f"Early stop at {epoch + 1} epoch!")
                    wandb_log({"epoch": epoch + 1, "Early stop": epoch + 1}, args)
                    all_metrics['early_stop_metric'].append(epoch)
                    break
            # val loss decreasing a lot
            else:
                best_val_loss_el = val_loss
                count = 0
        wandb_log({"epoch": epoch + 1, "patience count": count}, args)
        all_metrics['patience_metric'].append({epoch: count})

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch + 1, count, model, optimizer, scheduler,
                            best_val_loss, best_val_loss_el, args.save_dir, is_best=True)
            delete_oldest_checkpoint(args.save_dir)
            print(f'Best model saved with loss {best_val_loss}, at epoch {epoch + 1}')

        save_checkpoint(epoch + 1, count, model, optimizer, scheduler, best_val_loss, best_val_loss_el, args.save_dir)
        delete_oldest_checkpoint(args.save_dir)

        with open(args.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2)

    with open(args.metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GPT-2 model on cellular automaton data')
    parser.add_argument('--gpt2_size', type=str, default='small', help='the size of gpt2 model')
    parser.add_argument('--rule', type=int, required=True, help='Rule number for the cellular automaton')
    parser.add_argument('--k', type=int, required=True, help='Interval for outputting time points')
    parser.add_argument('--width', type=int, default=1000, help='Width of the cellular automaton')
    parser.add_argument('--sample_width', type=int, default=100, help='Width of the samples')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps for the cellular automaton')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=None, help='Micro batch size for training')
    parser.add_argument('--lr', type=float, default=2e-6, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs for training')
    parser.add_argument('--automation_type', type=str, default='random',
                        help='Type of cellular automaton creation to use')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to generate in each time t cellular automaton')
    parser.add_argument('--seed', type=int, default=1234, help='Sequence length for training')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index (0-5)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--wandb_project', type=str, default='cellular_automata')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_enable', type=bool, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_run_id', type=str, default=None)

    args = parser.parse_args()
    main(args)

# nohup python pretrain.py --gpt2_size xl --rule 150 --k 3 --micro_batch_size 8 --save_dir k3-256-01/rule150  --device 0 --wandb_enable False  > k3-256-01/train_150_3_xl.log 2>&1 &
