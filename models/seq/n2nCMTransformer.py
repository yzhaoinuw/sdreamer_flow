from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MultiHeadAttention
from layers.Freqtransform import STFT
from layers.patchEncoder import LinearPatchEncoder, LinearPatchEncoder2
from layers.transformer import Transformer, CrossAttnTransformer
from layers.norm import PreNorm
import librosa
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads 
        seq_len = args.seq_len
        dropout = args.dropout
        path_drop = args.path_drop
        e_layers = args.e_layers
        ca_layers = args.ca_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation
        n_sequences = args.n_sequences
        self.output_attentions = args.output_attentions
        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0 
        n_patches = seq_len // patch_len

        # self.stft_transform = STFT(win_length=patch_len,n_fft=256,hop_length=patch_len)
        self.eeg_transformer =  Transformer(
                                    patch_len, n_patches, e_layers, c_in, 
                                    inner_dim, n_heads = n_heads, d_head = d_head,
                                    dropout = dropout, path_drop = path_drop, 
                                    activation = activation, norm = norm_type, 
                                    mult = mult_ff, mix_type = args.mix_type, 
                                    cls = True, flag = "seq", domain = "time",
                                    output_attentions=self.output_attentions
                                )
        
        self.emg_transformer =  Transformer(
                                    patch_len, n_patches, e_layers, c_in, 
                                    inner_dim, n_heads = n_heads, d_head = d_head,
                                    dropout = dropout, path_drop = path_drop, 
                                    activation = activation, norm = norm_type, 
                                    mult = mult_ff, mix_type = args.mix_type, 
                                    cls = True, flag = "seq", domain = "time",
                                    output_attentions=self.output_attentions
                                )
        
        self.cm_transformer =  CrossAttnTransformer(
                                    ca_layers, 
                                    inner_dim, n_heads = n_heads, d_head = d_head,
                                    dropout = dropout, path_drop = path_drop, 
                                    activation = activation, norm = norm_type, 
                                    mult = mult_ff)
        
        self.seq_transformer =  Transformer(
                                    patch_len, n_sequences, e_layers, c_in, 
                                    inner_dim, n_heads = n_heads, d_head = d_head,
                                    dropout = dropout, path_drop = path_drop, 
                                    activation = activation, norm = norm_type, 
                                    mult = mult_ff, mix_type = args.mix_type, 
                                    cls = False, flag = "epoch", domain = "time",
                                    output_attentions=self.output_attentions
                                )

        self.proj = self._head = nn.Sequential( 
            nn.LayerNorm(inner_dim * n_traces),
            nn.Linear(inner_dim * n_traces, inner_dim)
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, c_out))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = x[:,:,0], x[:,:,1]

        eeg, eeg_attn = self.eeg_transformer(eeg)
        emg, emg_attn = self.emg_transformer(emg)

        cm_eeg, cm_emg, cm_eeg_attn , cm_emg_attn = self.cm_transformer(eeg, emg)

        cls_eeg, cls_emg = cm_eeg[:, :, -1], cm_emg[:, :, -1]
        # x_our --> [b, n, 2d]
        emb = torch.cat([cls_eeg, cls_emg], dim=-1)
        emb = self.proj(emb)
        emb, seq_attn = self.seq_transformer(emb)

        out = self.mlp_head(emb)
        out = rearrange(out, 'b e d -> (b e) d')
        emb = rearrange(emb, 'b e d -> (b e) d')
        label = rearrange(label, 'b e d -> (b e) d')

        out_dict = {
            "out": out,
            "seq_attn": seq_attn,
            "emb": emb, 
            "label": label
        }
        return out_dict