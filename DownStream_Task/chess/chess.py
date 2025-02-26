import os
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from itertools import chain
import sys
sys.path.append(".")  # Add project root to path

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from Pretrain.utils import (
    set_seed,
    get_lr_scheduler,
    create_gpt2_model,
    CustomGPT2Model,
)


def wandb_log(data, args):
    if args.wandb_enable:
        wandb.log(data)
    else:
        print(data)

# split 60 sequence length if nor enough fill them froim the begining
def split_subseq(san_list, seq_length=61):
    num_steps = len(san_list)
    subseqs = []
    for i in range(0, num_steps, seq_length):
        if i + seq_length >= num_steps:
            subseqs.append(san_list[-seq_length:])
            break
        else:
            subseqs.append(san_list[i:i+seq_length])
    return subseqs


class SanDataSet(Dataset):
    def __init__(self, data, san2id):
        self.data = data
        self.san2id = san2id
        # add new pad id
        self.pad_id = len(san2id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[:-1]
        y = sample[1:]
        x = [self.san2id[i] for i in x]
        y = [self.san2id[i] for i in y]
        num_pad = 60 - len(x)
        if num_pad > 0:
            x = x + [self.pad_id] * num_pad
            y = y + [-100] * num_pad
        return torch.tensor(x), torch.tensor(y)
    

def build_san_dataloader(san2id, args):
    df_train = pd.read_csv(args.train_data)
    df_val = pd.read_csv(args.val_data)
    print(df_train.shape, df_val.shape)
    
    train_data = list(chain(*[split_subseq(san.split()) for san in df_train['san'].values]))
    val_data = list(chain(*[split_subseq(san.split()) for san in df_val['san'].values]))
    print(len(train_data), len(val_data))
    
    train_set = SanDataSet(train_data, san2id)
    val_set = SanDataSet(val_data, san2id)
    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size)
    
    return train_loader, val_loader


class SanModel(nn.Module):
    def __init__(self, gpt2: CustomGPT2Model, vocab_size: int):
        """
        :param input_size: sample width
        :param gpt2:  CustomGPT2Model
        :param num_classes:
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, gpt2.input_projection.in_features)
        self.lm_head = nn.Linear(gpt2.output_layer.out_features, vocab_size)
        self.gpt2 = gpt2
        self.freeze()
        self.tie_weight()

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = self.gpt2(x, attention_mask)
        x = self.lm_head(x) # [b, l, v]
        return x

    def freeze(self):
        for param in self.gpt2.gpt2.parameters():
            param.requires_grad = False

    def tie_weight(self):
        self.lm_head.weight = nn.Parameter(self.embedding.weight.clone())


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, args):
    model.train()

    train_loss_his = []
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    total_iterations = len(train_loader) * gradient_accumulation_steps
    pbar = tqdm(total=total_iterations, desc=f"Training Epoch: {epoch}")

    vocab_size = model.lm_head.out_features

    for i, (X, Y) in enumerate(train_loader):
        X, Y = X.to(args.device), Y.to(args.device)
        mb_loss = 0
        for micro_x, micro_y in zip(torch.chunk(X, gradient_accumulation_steps), torch.chunk(Y, gradient_accumulation_steps)):
            logits = model(micro_x)
            loss = criterion(logits.reshape(-1, vocab_size), micro_y.reshape(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            mb_loss += loss.item()
            pbar.update(1)
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix({"loss": round(mb_loss, 9), "lr": round(scheduler.get_last_lr()[0], 9)})
        wandb.log({"train_loss": mb_loss, "learning_rate": scheduler.get_last_lr()[0]})

        train_loss_his.append(mb_loss)

    pbar.close()
    return train_loss_his
        


def validation_epoch(model, data_loader, criterion, args, mode="val"):
    model.eval()
    vocab_size = model.lm_head.out_features
    total_loss = 0.0
    total_samples = 0
    total_acc = 0.0
    with torch.no_grad():
        for i, (X, Y) in enumerate(tqdm(data_loader)):
            X, Y = X.to(args.device), Y.to(args.device)

            logits = model(X)
            loss = criterion(logits.reshape(-1, vocab_size), Y.reshape(-1))
            total_loss += loss.item()
            
            curr_result = (logits.argmax(dim=-1) == Y)[Y != -100]
            total_samples += curr_result.numel()
            total_acc += curr_result.sum().item()

            if mode == "train" and i >= 10:
                break

    avg_loss = total_loss / (i+1)
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc



def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, model_path, args):
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_val_loss_el = float("inf")
    patience = args.patience
    # assert patience == 4
    count = 0
    extra_counter = 0
    extra_training_triggered = False
    best_acc = 0
    best_val_acc = 0  # Initialize best validation accuracy

    for epoch in range(args.epochs):
        print(f"==================== epoch: {epoch + 1} / {args.epochs} ===========================")
        train_loss_his = train_epoch(model, train_dataloader, criterion, optimizer, scheduler, epoch, args)

        _, train_acc = validation_epoch(model, train_dataloader, criterion, args, mode="train")
        wandb.log({"epoch": epoch + 1, "train_acc": train_acc})
        val_loss, val_acc = validation_epoch(model, val_dataloader, criterion, args)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})
        print(train_acc, val_loss, val_acc)

        # Update best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.log({"epoch": epoch + 1, "best_val_acc": best_val_acc})

        if extra_training_triggered:
            if val_loss >= 1e-3:
                extra_training_triggered = False
                extra_counter = 0
            else:
                extra_counter += 1
                if extra_counter >= 50:
                    print(f"Early stop due to validation loss < 1e-3 and 50 extra epochs at {epoch + 1} epoch!")
                    wandb.log({"epoch": epoch + 1, "early_stop": epoch + 1})
                    break
        if val_loss < 1e-3 and not extra_training_triggered:
            extra_training_triggered = True


        # early stop
        if (epoch + 1) % 5 == 0:
            # val loss increasing or decreasing a few
            if val_loss - best_val_loss_el >= -1e-4:
                count += 1
                if count > patience:
                    print(f"Early stop at {epoch + 1} epoch!")
                    wandb.log({"epoch": epoch + 1, "early_stop": epoch + 1})
                    break
            # val loss decreasing a lot
            else:
                best_val_loss_el = val_loss
                count = 0
        wandb.log({"epoch": epoch + 1, "patience_count": count})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, model_path.split('/')[-1]))

    return best_acc, best_val_acc  # Return both best accuracy and best validation accuracy


def run_downstream(model_path, args):
    set_seed(args.seed)

    gpt2 = create_gpt2_model(args.gpt2_size, args.sample_width, args.seq_length)
    if model_path is not None and model_path != 'baseline':
        sd = torch.load(model_path, map_location='cpu')
        load_info = gpt2.load_state_dict(sd['model'])
        assert str(load_info) == "<All keys matched successfully>", "Failed to load model"

    san2id = json.load(open(args.label_path))
    san_model = SanModel(gpt2, len(san2id) + 1)  # special token
    san_model.to(args.device)

    print(f"Total trainable parameters: {sum(p.numel() for p in san_model.parameters() if p.requires_grad)}")

    train_dataloader, val_dataloader = build_san_dataloader(san2id, args)

    optimizer = torch.optim.Adam(san_model.parameters(), lr=args.lr)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=len(train_dataloader) * int(0.1 * args.epochs),
        total_steps=len(train_dataloader) * args.epochs,
    )
    

    best_acc, best_val_acc = train_model(san_model, train_dataloader, val_dataloader, optimizer, scheduler, model_path, args)

    return best_acc, best_val_acc


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    args.micro_batch_size = (args.batch_size if args.micro_batch_size is None else args.micro_batch_size)

    all_ckpt = []
    for _dir in args.ckpt_dirs.split(','):
        _dir = _dir.strip()
        all_ckpt += glob.glob(f"{_dir}/*.pth")
    if args.only_baseline:
        all_ckpt = ['baseline'] 

    result = []
    for ckpt in tqdm(all_ckpt):
        wandb.init(project="chess_san_downstream_1112",name=ckpt, config=args)
        best_acc, best_val_acc = run_downstream(ckpt, args)

        print(ckpt, best_acc, best_val_acc)
        curr_res = {
            'best_acc': best_acc,
            'best_val_acc': best_val_acc,
            'ckpt': ckpt
        }
        result.append(curr_res)
        pd.DataFrame(result).to_csv(os.path.join(args.save_dir, 'result_new_0925.csv'), index=False)
        wandb.finish()

    pd.DataFrame(result).to_csv(os.path.join(args.save_dir, 'result_new_0925.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tuning a CA model on downstream task')
    parser.add_argument('--gpt2_size', type=str, default='small', help='the siez of gpt2 model')
    parser.add_argument('--sample_width', type=int, default=100, help='Width of the samples')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for training')

    parser.add_argument('--train_data', type=str, default="DownStream_Task/chess/san-v2/san_train.csv")
    parser.add_argument('--val_data', type=str, default="DownStream_Task/chess/san-v2/san_val.csv")
    parser.add_argument('--label_path', type=str, default="DownStream_Task/chess/san-v2/san_label.json")

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=None, help='Micro batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='Sequence length for training')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index (0-5)')

    parser.add_argument('--ckpt_dirs', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--save_dir', type=str, default="sanouts-new_v2", help='Path to save the results')
    parser.add_argument("--patience", type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--wandb_enable', type=bool, default=True, help='Enable wandb logging')

    parser.add_argument("--only_baseline", action='store_true', help='Only run the baseline model')

    args = parser.parse_args()
    main(args)
