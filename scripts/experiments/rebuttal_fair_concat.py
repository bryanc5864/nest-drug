#!/usr/bin/env python3
"""Fair FiLM-vs-Concat comparison with jointly-trained projection.

Reviewers HVym, viq5, vUHL, rSJL all flagged that the original concat
baseline used a randomly-initialized, frozen projection. This script
trains FiLM and Concat variants under matched conditions: same backbone,
same data, same epochs, same optimizer; the only difference is fusion.
"""

import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.mpnn import MPNN
from src.models.context import ContextEmbedding
from src.training.data_utils import smiles_to_graph

DUDE_TARGETS = {
    'CHEMBL203': 'EGFR', 'CHEMBL217': 'DRD2', 'CHEMBL210': 'ADRB2',
    'CHEMBL4822': 'BACE1', 'CHEMBL206': 'ESR1', 'CHEMBL1937': 'HDAC2',
    'CHEMBL2971': 'JAK2', 'CHEMBL235': 'PPARG', 'CHEMBL340': 'CYP3A4',
    'CHEMBL244': 'FXA',
}


class FiLMHead(nn.Module):
    """FiLM fusion: gamma * h + beta."""
    def __init__(self, ctx_dim, feat_dim, hidden=256):
        super().__init__()
        self.gamma_net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.LayerNorm(hidden),
                                       nn.ReLU(), nn.Linear(hidden, feat_dim))
        self.beta_net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.LayerNorm(hidden),
                                      nn.ReLU(), nn.Linear(hidden, feat_dim))
        nn.init.zeros_(self.gamma_net[-1].weight); nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight);  nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, h, c):
        return self.gamma_net(c) * h + self.beta_net(c)


class ConcatHead(nn.Module):
    """Concat fusion with jointly-trained projection MLP.

    h_mod = MLP([h || c]) -> feat_dim
    """
    def __init__(self, ctx_dim, feat_dim, hidden=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + ctx_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, h, c):
        return self.proj(torch.cat([h, c], dim=-1))


class AdditiveHead(nn.Module):
    """Additive fusion: h + beta(c)."""
    def __init__(self, ctx_dim, feat_dim, hidden=256):
        super().__init__()
        self.beta_net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.LayerNorm(hidden),
                                      nn.ReLU(), nn.Linear(hidden, feat_dim))
        nn.init.zeros_(self.beta_net[-1].weight); nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, h, c):
        return h + self.beta_net(c)


class GatedConcatHead(nn.Module):
    """Gated concat: a stronger concat baseline using a gating mechanism.

    h_mod = h * sigmoid(W [h||c]) + Linear([h||c])
    Equivalent capacity to FiLM but starting from concat.
    """
    def __init__(self, ctx_dim, feat_dim, hidden=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim + ctx_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Linear(hidden, feat_dim), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + ctx_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Linear(hidden, feat_dim),
        )

    def forward(self, h, c):
        cat = torch.cat([h, c], dim=-1)
        return h * self.gate(cat) + self.proj(cat)


class CondModel(nn.Module):
    def __init__(self, fusion: str, num_programs: int, ctx_dim=128, feat_dim=512):
        super().__init__()
        # Match repo MPNN signature: 69 atom feats, 9 bond feats
        self.mpnn = MPNN(node_input_dim=69, edge_input_dim=9, hidden_dim=256,
                         num_layers=4, dropout=0.1)
        self.ctx = ContextEmbedding(num_programs, ctx_dim, init_std=0.02)
        if fusion == 'film':
            self.head = FiLMHead(ctx_dim, feat_dim)
        elif fusion == 'concat':
            self.head = ConcatHead(ctx_dim, feat_dim)
        elif fusion == 'additive':
            self.head = AdditiveHead(ctx_dim, feat_dim)
        elif fusion == 'gated':
            self.head = GatedConcatHead(ctx_dim, feat_dim)
        elif fusion == 'none':
            self.head = None
        else:
            raise ValueError(fusion)
        self.fusion = fusion
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, nf, ei, ef, b, pid):
        h = self.mpnn(nf, ei, ef, b)
        if self.head is not None:
            c = self.ctx(pid)
            h = self.head(h, c)
        return self.predictor(h).squeeze(-1)


class MolDataset(Dataset):
    def __init__(self, df, prog_map):
        self.df = df.reset_index(drop=True)
        self.prog_map = prog_map

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = smiles_to_graph(r['smiles'])
        if g is None:
            return None
        pid = self.prog_map.get(r['target_chembl_id'], 0)
        return {
            'g': g,
            'y': float(r['pchembl_median']),
            'pid': int(pid),
            'tgt': r['target_chembl_id'],
        }


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    nf, ei, ef, bi = [], [], [], []
    off = 0
    for i, b in enumerate(batch):
        g = b['g']
        nf.append(g['node_features'])
        ei.append(g['edge_index'] + off)
        ef.append(g['edge_features'])
        bi.extend([i] * g['num_atoms'])
        off += g['num_atoms']
    return {
        'nf': torch.cat(nf), 'ei': torch.cat(ei, dim=1), 'ef': torch.cat(ef),
        'b': torch.tensor(bi), 'y': torch.tensor([b['y'] for b in batch]),
        'pid': torch.tensor([b['pid'] for b in batch]),
        'tgt': [b['tgt'] for b in batch],
    }


def train_one(fusion, train_df, val_df, prog_map, device, epochs=3, batch=256, lr=3e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CondModel(fusion, num_programs=len(prog_map)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_df) // batch)

    train_ds = MolDataset(train_df, prog_map)
    val_ds = MolDataset(val_df, prog_map)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            collate_fn=collate, num_workers=2)
    crit = nn.MSELoss()
    print(f'[{fusion}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    for ep in range(epochs):
        model.train()
        loss_acc, n = 0, 0
        for b in train_loader:
            if b is None: continue
            opt.zero_grad()
            pred = model(b['nf'].to(device), b['ei'].to(device), b['ef'].to(device),
                         b['b'].to(device), b['pid'].to(device))
            loss = crit(pred, b['y'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            loss_acc += loss.item() * len(pred); n += len(pred)
        train_mse = loss_acc / max(n, 1)

        # Val
        model.eval()
        all_p, all_y, all_t = [], [], []
        with torch.no_grad():
            for b in val_loader:
                if b is None: continue
                pred = model(b['nf'].to(device), b['ei'].to(device), b['ef'].to(device),
                             b['b'].to(device), b['pid'].to(device))
                all_p.append(pred.cpu().numpy())
                all_y.append(b['y'].numpy())
                all_t += b['tgt']
        p = np.concatenate(all_p); y = np.concatenate(all_y)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        # AUC: above/below per-target median
        med = np.median(y)
        bin_y = (y > med).astype(int)
        try:
            auc = roc_auc_score(bin_y, p)
        except Exception:
            auc = 0.5
        print(f'  ep{ep+1}/{epochs}  train_mse={train_mse:.4f}  val_rmse={rmse:.4f}  val_auc={auc:.4f}')

    # Per-target AUC on val
    per_target = {}
    df = pd.DataFrame({'pred': p, 'y': y, 'tgt': all_t})
    for tgt in df['tgt'].unique():
        sub = df[df['tgt'] == tgt]
        if len(sub) < 20: continue
        m = sub['y'].median()
        bin_y = (sub['y'] > m).astype(int)
        if bin_y.nunique() < 2: continue
        try:
            per_target[tgt] = float(roc_auc_score(bin_y, sub['pred']))
        except Exception:
            pass
    return {'fusion': fusion, 'rmse': rmse, 'auc': float(auc), 'per_target_auc': per_target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=80000)
    ap.add_argument('--n-val', type=int, default=10000)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--fusions', nargs='+', default=['none', 'concat', 'additive', 'gated', 'film'])
    ap.add_argument('--out', type=str, default='/tmp/nest_rebuttal/fair_concat.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    train = pd.read_parquet('/tmp/nest_train_split.parquet')
    val = pd.read_parquet('/tmp/nest_val_split.parquet')

    # Focus on DUD-E targets + a sample of others to keep program count reasonable
    keep = list(DUDE_TARGETS.keys())
    extra = train['target_chembl_id'].value_counts().head(50).index.tolist()
    keep = list(dict.fromkeys(keep + extra))
    train = train[train['target_chembl_id'].isin(keep)].copy()
    val = val[val['target_chembl_id'].isin(keep)].copy()
    train = train.dropna(subset=['pchembl_median'])
    val = val.dropna(subset=['pchembl_median'])

    prog_map = {t: i for i, t in enumerate(sorted(keep))}
    print(f'targets={len(keep)}  train_pool={len(train)}  val_pool={len(val)}')

    # Stratified subsample
    train = train.groupby('target_chembl_id', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(args.n_train // len(keep), 100)), random_state=42)
    )
    val = val.groupby('target_chembl_id', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(args.n_val // len(keep), 20)), random_state=42)
    )
    print(f'after subsample  train={len(train)}  val={len(val)}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    all_results = []
    for fusion in args.fusions:
        seed_results = []
        for seed in args.seeds:
            print(f'\n=== {fusion}  seed={seed} ===')
            r = train_one(fusion, train, val, prog_map, device,
                          epochs=args.epochs, batch=args.batch, seed=seed)
            r['seed'] = seed
            seed_results.append(r)
        # Aggregate
        aucs = [r['auc'] for r in seed_results]
        rmses = [r['rmse'] for r in seed_results]
        agg = {
            'fusion': fusion,
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
            'seeds': seed_results,
        }
        # Per-target mean
        all_targets = set()
        for r in seed_results: all_targets |= set(r['per_target_auc'].keys())
        per_target = {}
        for t in all_targets:
            vals = [r['per_target_auc'][t] for r in seed_results if t in r['per_target_auc']]
            per_target[t] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
        agg['per_target'] = per_target
        all_results.append(agg)

    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'prog_map': prog_map, 'results': all_results}, f, indent=2)
    print(f'\nSaved {args.out}')

    print('\n=== SUMMARY ===')
    for r in all_results:
        print(f"{r['fusion']:>10}  AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"RMSE = {r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}")


if __name__ == '__main__':
    main()
