import glob
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import os
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
import sys
sys.path.append(".")  # Add project root to path

from Pretrain.utils import (
    set_seed,
    create_gpt2_model,
    create_attention_mask,
    RGBModel,
)

COLORS = ['B', 'P', 'R', 'G']
COLOR2ID = {'B': 1, 'P': 2, 'R': 3, 'G': 4}
ID2COLOR = {1: 'B', 2: 'P', 3: 'R', 4: 'G'}


def build_l2_dataloader(batch_size, sample_path='downstream_data_level2_H_1103.pth'):
    l2_data = torch.load(sample_path)
    train_samples = l2_data['train']
    val_samples = l2_data['val']
    train_dataset = TensorDataset(
        train_samples[:, :60, :].float(),
        train_samples[:, 1:, :].long(),
    )
    val_dataset = TensorDataset(
        val_samples[:, :60, :].float(),
        val_samples[:, 1:, :].long(),
    )
    print(f"Number samples:\nTrain:{len(train_dataset)}\nVal:{len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, thresholds_list=None, epochs=100, lr=1e-3, num_classes=5, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if thresholds_list is None:
        thresholds_list = [80, 90]
    reach_list = [-1] * len(thresholds_list)
    threshold_reached = {t: False for t in thresholds_list}

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model = None

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for batch_sequences, batch_targets in train_loader:
            batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
            attention_mask = create_attention_mask(
                batch_sequences.size(1)
            ).repeat(batch_sequences.size(0), 1, 1, 1).to(device)

            optimizer.zero_grad()
            logits = model(batch_sequences, attention_mask)
            loss = criterion(logits.reshape(-1, num_classes), batch_targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_acc_number = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
                attention_mask = create_attention_mask(batch_sequences.size(1)).repeat(batch_sequences.size(0), 1, 1, 1).to(device)
                logits = model(batch_sequences, attention_mask)
                loss = criterion(logits.reshape(-1, num_classes), batch_targets.reshape(-1))
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_acc_number += ((preds == batch_targets).float().sum(-1) == 100).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_acc_number / len(val_loader.dataset) / 60
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        for i, threshold in enumerate(thresholds_list):
            if val_acc * 100 >= threshold and not threshold_reached[threshold]:
                reach_list[i] = epoch + 1
                threshold_reached[threshold] = True
                wandb.log({
                    f"threshold_{threshold}_reached": epoch + 1,
                    f"epoch_threshold_{threshold}": epoch + 1
                })
                print(f"Threshold {threshold}% reached at epoch {epoch + 1}")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
            print(f"New best model found at epoch {epoch}")
            wandb.log({
                "best_val_accuracy": best_val_acc, 
                "best_val_loss": best_val_loss,
                "best_model_epoch": epoch
            })
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

    if best_model is not None:
        model.load_state_dict(best_model)
    else:
        print("Warning: No best model was saved during training.")

    for threshold, epoch in zip(thresholds_list, reach_list):
        if epoch != -1:
            wandb.run.summary[f"final_epoch_threshold_{threshold}"] = epoch
        else:
            wandb.run.summary[f"final_epoch_threshold_{threshold}"] = "Not Reached"

    return reach_list, model, best_val_acc


def run_downstream(gpt2_size, sample_width, seq_length, device, thresholds_list, batch_size,
                   model_path=None, epochs=1000, seed=42, patience=30):
    print(f"seed: {seed}")
    set_seed(seed)

    gpt2 = create_gpt2_model(gpt2_size, sample_width, seq_length)
    if model_path is not None and model_path != 'baseline':
        load_info = gpt2.load_state_dict(torch.load(model_path, map_location='cpu')["model"])
        assert str(load_info) == "<All keys matched successfully>", "Failed to load model"
    rgb_model = RGBModel(sample_width, gpt2, num_classes=len(COLORS)+1)

    train_dataloader, val_dataloader = build_l2_dataloader(batch_size, args.data_path)

    wandb.init(project="ARC_hard1116_grad", config={
        "gpt2_size": gpt2_size,
        "sample_width": sample_width,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "model_path": model_path,
        "patience": patience
    },name = model_path,reinit=True)
    
    reach_list, best_model, best_val_acc = train_model(rgb_model, train_dataloader, val_dataloader,
                             epochs=epochs, device=device, thresholds_list=thresholds_list, 
                             num_classes=len(COLORS)+1, patience=patience)

    wandb.finish()
    return reach_list, best_model, best_val_acc

def main(args):
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    all_ckpt = glob.glob(f"{args.ckpt_dirs}/best_*.pth")
    if args.only_baseline:
        all_ckpt = ['baseline'] 

    print(all_ckpt)
    thresholds_list = [int(_.strip()) for _ in args.thresholds.split(',')]

    result = []
    for ckpt in tqdm(all_ckpt):
        curr_reach_list, best_model, best_val_acc = run_downstream(args.gpt2_size, args.sample_width, args.seq_length,
                                         args.device, thresholds_list, args.batch_size,
                                         model_path=ckpt, seed=args.seed, epochs=args.epochs, patience=args.patience)
        print(ckpt, curr_reach_list)
        curr_res = {f"thres_{t}": r for t, r in zip(thresholds_list, curr_reach_list)}
        curr_res['ckpt'] = ckpt
        curr_res['best_accuracy'] = best_val_acc
        result.append(curr_res)

        results_save_path = os.path.join(args.save_dir, f"results_{ckpt.split('/')[-1]}.csv")
        pd.DataFrame(result).to_csv(results_save_path, index=False)

        model_save_path = os.path.join(args.save_dir, f"best_model_{ckpt.split('/')[-1]}")
        torch.save(best_model.state_dict(), model_save_path)

    results_save_path = os.path.join(args.save_dir, f"results_{ckpt.split('/')[-1]}.csv")
    pd.DataFrame(result).to_csv(results_save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tuning a CA model on downstream task')
    parser.add_argument('--gpt2_size', type=str, default='small', help='the size of gpt2 model')
    parser.add_argument('--sample_width', type=int, default=100, help='Width of the samples')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index (0-5)')
    parser.add_argument('--thresholds', type=str, default='10,20,30,40,50,60,70,80,90,100')
    parser.add_argument('--ckpt_dirs', type=str, default='./')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the results')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument("--only_baseline", action='store_true', help='Only run the baseline model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    args = parser.parse_args()
    main(args)