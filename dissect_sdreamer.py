# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:06:26 2024

@author: yzhao
"""

import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from models.seq import n2nSeqNewMoE2
from utils.preprocessing import reshape_sleep_data


class SequenceDataset(Dataset):
    def __init__(self, normalized_sleep_data):
        self.traces = normalized_sleep_data

    def __len__(self):
        return self.traces.shape[0]

    def __getitem__(self, idx):
        trace = self.traces[idx]
        return trace


def make_dataset(data: dict, n_sequences: int = 64):
    eeg, emg = reshape_sleep_data(data, has_labels=False)
    sleep_data = np.stack((eeg, emg), axis=1)
    sleep_data = torch.from_numpy(sleep_data)
    sleep_data = torch.unsqueeze(sleep_data, dim=2)  # shape [n_seconds, 2, 1, seq_len]
    mean, std = torch.mean(sleep_data, dim=0), torch.std(sleep_data, dim=0)
    normalized_data = (sleep_data - mean) / std

    n_seconds = normalized_data.shape[0]
    n_to_crop = n_seconds % n_sequences
    if n_to_crop != 0:
        normalized_data = torch.cat(
            [normalized_data[:-n_to_crop], normalized_data[-n_sequences:]], dim=0
        )

    normalized_data = normalized_data.reshape(
        (
            -1,
            n_sequences,
            normalized_data.shape[1],
            normalized_data.shape[2],
            normalized_data.shape[3],
        )
    )
    dataset = SequenceDataset(normalized_data)
    return dataset, n_seconds, n_to_crop


# %%

config = dict(
    model="SeqNewMoE2",
    data="Seq",
    isNE=False,
    features="ALL",
    n_sequences=64,
    useNorm=True,
    seq_len=512,
    patch_len=16,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=2,
    ca_layers=1,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    activation="glu",
    norm_type="layernorm",
    output_attentions=False,
)


def build_args(**kwargs):
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)

    for k, v in config.items():
        parser_dict[k] = v
    for k, v in kwargs.items():
        parser_dict[k] = v
    return args


# %%
def infer(data, checkpoint_path, batch_size=32):
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = n2nSeqNewMoE2.Model(args)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    n_sequences = config["n_sequences"]
    dataset, n_seconds, n_to_crop = make_dataset(data)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        all_pred = []
        all_prob = []

        with tqdm(total=n_seconds, unit=" seconds of signal") as pbar:
            for batch, traces in enumerate(data_loader, 1):
                traces = traces.to(device)  # [batch_size, 64, 2, 1, 512]
                out_dict = model(traces, label=None)
                out = out_dict["out"]

                prob = torch.max(torch.softmax(out, dim=1), dim=1).values
                all_prob.append(prob.detach().cpu())

                pred = np.argmax(out.detach().cpu(), axis=1)
                # pred = out_dict["predictions"]
                all_pred.append(pred)

                pbar.update(batch_size * n_sequences)
            pbar.set_postfix({"Number of batches": batch})

        if n_to_crop != 0:
            all_pred[-1] = torch.cat(
                (
                    all_pred[-1][: -args.n_sequences],
                    all_pred[-1][-args.n_sequences :][-n_to_crop:],
                )
            )
            all_prob[-1] = torch.cat(
                (
                    all_prob[-1][: -args.n_sequences],
                    all_prob[-1][-args.n_sequences :][-n_to_crop:],
                )
            )

        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)

    return all_pred, all_prob


# %%

from scipy.io import loadmat

checkpoint_path = "C:/Users/yzhao/python_projects/sleep_scoring/models/sdreamer/checkpoints/SeqNewMoE2_Seq_ftALL_pl16_ns64_dm128_el2_dff512_eb0_scale0.0_bs64_f1_augment_10.pth.tar"
mat_file = (
    "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/115_gs.mat"
)
data = loadmat(mat_file)

args = build_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = n2nSeqNewMoE2.Model(args)
model = model.to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["state_dict"])

batch_size=32
n_sequences = config["n_sequences"]
dataset, n_seconds, n_to_crop = make_dataset(data)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

model.eval()

#%%
from einops.layers.torch import Rearrange


with torch.no_grad():
    all_pred = []
    all_prob = []

    traces = next(iter(data_loader))
    traces = traces.to(device)  # [batch_size, 64, 2, 1, 512]
    out_dict = model(traces, label=None)
    out = out_dict["out"]

    prob = torch.max(torch.softmax(out, dim=1), dim=1).values
    all_prob.append(prob.detach().cpu())

    pred = np.argmax(out.detach().cpu(), axis=1)
    # pred = out_dict["predictions"]
    all_pred.append(pred)
    
#%%

    encoder = model.eeg_transformer.patch_ecncoder
    traces.shape # torch.Size([32, 64, 2, 1, 512])
    eeg, emg = traces[:, :, 0], traces[:, :, 1]
    eeg.shape # torch.Size([32, 64, 1, 512])
    
    x = encoder(eeg)
    x.shape # torch.Size([32, 64, 32, 128])
    rearrange = Rearrange("b ... c (n l) -> b ... n (l c)", l=16)
    h = rearrange(eeg)
    h.shape # torch.Size([32, 64, 32, 16])
    
    get_cls = model.eeg_transformer.get_cls
    x_cls = get_cls(x)
    x_cls.shape # torch.Size([32, 64, 33, 128])
    
    eeg_transformer = model.eeg_transformer
    emg_transformer = model.emg_transformer
    eeg_out = eeg_transformer(eeg)[0]
    emg_out = emg_transformer(emg)[0]
    cls_eeg = eeg_out[:, :, -1]
    cls_emg = emg_out[:, :, -1]
    
    eeg_out.shape # [32, 64, 33, 128]
    cls_eeg.shape # [32, 64, 128]
    
#%%

    moe_transformer = model.moe_transformer
    eeg_loader = model.moe_transformer.eeg_loader
    emg_loader = model.moe_transformer.emg_loader
    eeg_embs, eeg_mask = eeg_loader(cls_eeg)
    emg_embs, emg_mask = emg_loader(cls_emg)
    eeg_embs.shape # torch.Size([32, 64, 128])
    eeg_mask.shape # torch.Size([32, 64])
    
    # add sort of token type embegging, does not change shape
    eeg_embs, emg_embs = (
        eeg_embs + moe_transformer.mod_emb(torch.full_like(eeg_mask, 0)),
        emg_embs + moe_transformer.mod_emb(torch.full_like(emg_mask, 1)),
    )
    eeg_embs.shape # torch.Size([32, 64, 128])
#%%    
    co_embeds = torch.cat([eeg_embs, emg_embs], dim=1)
    co_masks = torch.cat([eeg_mask, emg_mask], dim=1)
    co_embeds.shape # torch.Size([32, 128, 128])
    
    # dissect MoEBlock
    attention = moe_transformer.transformer[0].attn
    atten_out = attention(co_embeds)
    atten_out.shape # torch.Size([32, 128, 128])
    
    x = co_embeds
    for i, blk in enumerate(moe_transformer.transformer):
        x = blk(x, mask=co_masks, modality_type="mix")

    x = moe_transformer.norm(x)
    cls_feats = moe_transformer.pool(x)
    logits = model.cls_head(cls_feats)
    logits.shape # torch.Size([32, 64, 3])
