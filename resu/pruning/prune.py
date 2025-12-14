# ============================
#  CHUNK 6A — CORE INFRA
#  Qwen2.5 FP16 Wanda++ + DSNoT
# ============================

import torch
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset


# ----------------------------
# Config
# ----------------------------
@dataclass
class PruneConfig:
    nsamples: int = 128
    seed: int = 0
    dataset: str = "c4"
    sparsity: float = 0.5
    wanda_alpha: float = 100.0

    dsnot_gamma: float = 1.0
    dsnot_cycles: int = 50
    dsnot_update_eps: float = 0.01

    ro_iters: int = 5
    ro_lr: float = 3e-7
    ro_subset: int = 32


# ----------------------------
# Dataset loader
# ----------------------------
def get_loaders(dataset_name, nsamples, seed, seqlen, tokenizer):
    torch.manual_seed(seed)

    if dataset_name.lower() == "c4":
        d = load_dataset("c4", "en", split="train", streaming=True)
    else:
        raise ValueError("unknown dataset")

    it = iter(d)
    arr = []
    while len(arr) < nsamples:
        ex = next(it)
        tx = ex["text"].strip()
        if tx == "":
            continue
        ids = tokenizer(
            tx,
            truncation=True,
            max_length=seqlen,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"][0]
        arr.append(ids)
    return torch.stack(arr), None


# ----------------------------
# Prepare calibration input
# ----------------------------
def prepare_calibration_input(model, dataloader, device):
    model.eval()
    dataloader = dataloader.to(device)

    ns, sl = dataloader.shape
    am = torch.ones_like(dataloader, device=device)
    pos = torch.arange(sl, device=device).unsqueeze(0).repeat(ns, 1)

    if hasattr(model.model, "embed_tokens"):
        emb = model.model.embed_tokens
    else:
        emb = model.embed_tokens

    X = emb(dataloader).to(device)
    Y = torch.zeros_like(X, device=device)
    return X, Y, am, pos


# ----------------------------
# Qwen2.5 linear layer finder
# ----------------------------
def find_layers_qwen2(layer):
    out = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear):
            out[name] = mod
    return out


# ----------------------------
# Wanda++ wrapper
# ----------------------------
class WandaPlusWrapper:
    def __init__(self, layer, alpha):
        self.layer = layer
        self.alpha = alpha
        W = layer.weight.data
        self.rows, self.cols = W.shape
        dev = W.device
        self.X_norm = torch.zeros(self.cols, device=dev)
        self.grad_sq = torch.zeros_like(W, device=dev)
        self.nsamples = 0

    def add_acts(self, inp):
        # inp: (1, seq, in)
        x = inp.reshape(-1, inp.shape[-1]).t()
        self.X_norm += torch.norm(x, p=2, dim=1) ** 2
        self.nsamples += 1

    def add_grad(self, g):
        self.grad_sq += g.pow(2)

    def score(self):
        xn = self.X_norm / max(1, self.nsamples)
        gs = self.grad_sq / max(1, self.nsamples)
        W = self.layer.weight.data.abs()
        return W * (self.alpha * gs + xn.reshape(1, -1))

    def free(self):
        self.X_norm = None
        self.grad_sq = None
        torch.cuda.empty_cache()


# ----------------------------
# Regional gradient calc
# ----------------------------
def compute_regional_gradients(block, X, wrappers, device, micro=4):
    for w in wrappers.values():
        w.layer.weight.requires_grad = True

    ns = X.shape[0]
    for i in range(0, ns, micro):
        xb = X[i:i+micro].detach().to(device)
        xb.requires_grad = False

        out = block(xb)[0]
        loss = torch.norm(out, p=2)

        block.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, w in wrappers.items():
                g = w.layer.weight.grad
                if g is not None:
                    w.add_grad(g)

        block.zero_grad()
        del out, loss, xb
        torch.cuda.empty_cache()

    for w in wrappers.values():
        w.layer.weight.requires_grad = False


# ----------------------------
# Wanda++ metric extraction
# ----------------------------
def run_wanda_plus(model, tokenizer, cfg, device):
    dl, _ = get_loaders(cfg.dataset, cfg.nsamples, cfg.seed, model.seqlen, tokenizer)

    with torch.no_grad():
        inps, outs, am, pos = prepare_calibration_input(model, dl, device)

    layers = model.model.layers
    stats = {}

    for li, block in enumerate(layers):
        subset = find_layers_qwen2(block)

        inps = inps.to(device)
        outs = outs.to(device)
        am = am.to(device)
        pos = pos.to(device)

        wrappers = {n: WandaPlusWrapper(subset[n], cfg.wanda_alpha) for n in subset}

        # hooks
        def hook_maker(n):
            def h(m, inp, out):
                wrappers[n].add_acts(inp[0].data)
            return h

        hlist = []
        for n, mod in subset.items():
            hlist.append(mod.register_forward_hook(hook_maker(n)))

        for j in range(cfg.nsamples):
            with torch.no_grad():
                outs[j] = block(inps[j].unsqueeze(0), attention_mask=am, position_ids=pos)[0]

        for h in hlist:
            h.remove()

        compute_regional_gradients(block, inps, wrappers, device)

        layer_stats = {}
        for n, w in wrappers.items():
            W = subset[n].weight.data.clone().cpu()
            sc = w.score().detach().cpu()
            layer_stats[n] = {
                "weight": W,
                "score": sc,
                "X_norm": (w.X_norm / max(1, w.nsamples)).detach().cpu(),
                "grad_sq": (w.grad_sq / max(1, w.nsamples)).detach().cpu()
            }
            w.free()

        stats[li] = layer_stats

        for j in range(cfg.nsamples):
            with torch.no_grad():
                outs[j] = block(inps[j].unsqueeze(0), attention_mask=am, position_ids=pos)[0]

        inps, outs = outs, inps
        torch.cuda.empty_cache()

    return stats


# ============================
#  CHUNK 6B — DSNoT + RO + PIPELINE
# ============================

import torch
import torch.nn as nn


# ----------------------------
# DSNoT — variance-aware mask refinement
# ----------------------------
def dsnot_refine_mask(W, score_init, X, Y_dense, sparsity, gamma, max_cycles, update_eps):
    device = W.device
    out_dim, in_dim = W.shape
    N = X.shape[0]

    X_mean = X.mean(dim=0)
    X_var = X.var(dim=0, unbiased=True) + 1e-6

    flat = score_init.view(-1)
    k = int(sparsity * flat.numel())
    if k > 0:
        thr, _ = torch.kthvalue(flat, k)
        prune_mask = (score_init <= thr)
    else:
        prune_mask = torch.zeros_like(score_init, dtype=torch.bool)
    mask = ~prune_mask

    W_sparse = W * mask
    Y_sparse = X @ W_sparse.t()
    err = Y_dense - Y_sparse
    err_mean = err.mean(dim=0, keepdim=True).t()

    W_orig = W.clone()
    X_mean_b = X_mean.view(1, -1)
    X_var_pow = X_var.pow(gamma).view(1, -1)

    cyc = 0
    while cyc < max_cycles:
        cyc += 1

        sign_e = torch.sign(err_mean).view(-1, 1)

        grow_numer = sign_e * W_orig * X_mean_b
        grow_score = grow_numer / X_var_pow
        grow_score[mask] = -1e30

        grow_idx = grow_score.argmax(dim=1)

        prune_score = score_init.clone()
        prune_score[~mask] = 1e30
        prune_idx = prune_score.argmin(dim=1)

        regW = W_orig[torch.arange(out_dim), grow_idx]
        regX = X[:, grow_idx]
        reg_term = regX * regW

        prW = W_orig[torch.arange(out_dim), prune_idx]
        prX = X[:, prune_idx]
        pr_term = prX * prW

        new_err = err + pr_term - reg_term
        new_em = new_err.mean(dim=0, keepdim=True).t()

        diff = (new_em - err_mean).abs().mean()
        if diff < update_eps:
            break

        mask[torch.arange(out_dim), prune_idx] = False
        mask[torch.arange(out_dim), grow_idx] = True

        err = new_err
        err_mean = new_em

    W_final = W_orig * mask
    Y_sparse_final = X @ W_final.t()
    return mask, Y_sparse_final


# ----------------------------
# Regional Optimization
# ----------------------------
def apply_ro(block, mask_dict, X_list, Y_list, am, pos, device, iters, lr, ss):
    N = len(X_list)
    ss = min(ss, N)

    X = torch.stack(X_list[:ss], dim=0).squeeze(1).to(device)
    T = torch.stack(Y_list[:ss], dim=0).squeeze(1).to(device)

    opt = torch.optim.RMSprop(block.parameters(), lr=lr)
    block.train()

    prunable = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and name in mask_dict:
            prunable[name] = mask_dict[name].to(device)

    for _ in range(iters):
        opt.zero_grad()

        Y = block(X, attention_mask=am, position_ids=pos)[0]
        loss = nn.functional.mse_loss(Y, T)
        loss.backward()

        with torch.no_grad():
            for name, mod in block.named_modules():
                if isinstance(mod, nn.Linear) and name in prunable:
                    if mod.weight.grad is not None:
                        mod.weight.grad *= prunable[name]

        opt.step()

        with torch.no_grad():
            for name, mod in block.named_modules():
                if isinstance(mod, nn.Linear) and name in prunable:
                    mod.weight.data *= prunable[name]

        torch.cuda.empty_cache()

    return block


# ----------------------------
# DSNoT full layer-by-layer refinement
# ----------------------------
def run_dsnot(model, wanda_stats, calib_inps, calib_outs, am, pos, cfg, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    L = len(layers)

    final_masks = {}
    inps = calib_inps
    outs = calib_outs

    for li in range(L):
        block = layers[li]
        subset = find_layers_qwen2(block)

        dev = device
        inps = inps.to(dev)
        outs = outs.to(dev)
        am = am.to(dev)
        pos = pos.to(dev)

        final_masks[li] = {}

        Y_dense = torch.stack(outs, dim=0).squeeze(1).to(dev)
        Xmat = inps.reshape(-1, inps.shape[-1]).to(dev)

        for name in subset:
            st = wanda_stats[li][name]
            W = st["weight"].to(dev)
            score = st["score"].to(dev)

            mask, _ = dsnot_refine_mask(
                W=W,
                score_init=score,
                X=Xmat,
                Y_dense=Y_dense,
                sparsity=cfg.sparsity,
                gamma=cfg.dsnot_gamma,
                max_cycles=cfg.dsnot_cycles,
                update_eps=cfg.dsnot_update_eps
            )

            final_masks[li][name] = mask.cpu()
            subset[name].weight.data = subset[name].weight.data.to(dev) * mask

        apply_ro(
            block,
            final_masks[li],
            inps,
            outs,
            am,
            pos,
            dev,
            cfg.ro_iters,
            cfg.ro_lr,
            cfg.ro_subset
        )

        next_outs = []
        for j in range(len(inps)):
            with torch.no_grad():
                o = block(
                    inps[j].unsqueeze(0),
                    attention_mask=am,
                    position_ids=pos
                )[0]
                next_outs.append(o)

        outs, inps = next_outs, outs
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return final_masks


# ----------------------------
# Sparsity checker
# ----------------------------
def check_sparsity(model):
    L = model.model.layers
    tot = 0
    z = 0
    for b in L:
        subs = find_layers_qwen2(b)
        for _, m in subs.items():
            W = m.weight.data
            tot += W.numel()
            z += (W == 0).sum().item()
    print("final sparsity:", z / tot)


# ----------------------------
# Main pipeline
# ----------------------------
def prune_qwen2_wandaplus_dsnot(model, tokenizer, cfg, device):
    dl, _ = get_loaders(cfg.dataset, cfg.nsamples, cfg.seed, model.seqlen, tokenizer)

    with torch.no_grad():
        inps, outs, am, pos = prepare_calibration_input(model, dl, device)

    wanda_stats = run_wanda_plus(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device
    )

    masks = run_dsnot(
        model=model,
        wanda_stats=wanda_stats,
        calib_inps=inps,
        calib_outs=outs,
        am=am,
        pos=pos,
        cfg=cfg,
        device=device
    )

    check_sparsity(model)
    return model, masks



# ===========================================
#  CHUNK 9 — DEBUGGING / PROFILING SUITE
#  For Wanda++ + DSNoT + RO (Qwen2.5 FP16)
# ===========================================

import torch
import time
import psutil
import gc


# ------------------------------------------------
# 1. Timer Utility
# ------------------------------------------------
class Timer:
    def __init__(self, name="block"):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        dt = time.time() - self.t0
        print(f"[TIMER] {self.name}: {dt:.3f}s")


# ------------------------------------------------
# 2. GPU Memory Tracker
# ------------------------------------------------
class MemTracker:
    def __init__(self, label=""):
        self.label = label

    def __call__(self):
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[MEM] {self.label} allocated={alloc:.1f}MB reserved={reserved:.1f}MB")


def cpu_mem():
    p = psutil.Process()
    rss = p.memory_info().rss / 1024**2
    print(f"[CPU MEM] {rss:.1f}MB")


# ------------------------------------------------
# 3. Activation Distribution Printer
# ------------------------------------------------
def inspect_activations(X, name="X"):
    print(f"[ACT] {name} shape={tuple(X.shape)}")
    print(f"[ACT] {name} mean={X.mean().item():.4f}")
    print(f"[ACT] {name} std={X.std().item():.4f}")
    print(f"[ACT] {name} max={X.max().item():.4f}")
    print(f"[ACT] {name} min={X.min().item():.4f}")


# ------------------------------------------------
# 4. Score Statistics (Wanda++ and DSNoT)
# ------------------------------------------------
def inspect_scores(score, name="score"):
    print(f"[SCORE] {name} mean={score.mean().item():.4f} max={score.max().item():.4f} min={score.min().item():.4f}")
    nz = (score > 0).sum().item()
    print(f"[SCORE] {name} nonzero={nz}/{score.numel()}")


def inspect_mask(mask, name="mask"):
    k = mask.sum().item()
    print(f"[MASK] {name}: kept={k}/{mask.numel()} ({k/mask.numel():.4f})")


# ------------------------------------------------
# 5. FLOP Estimator
# ------------------------------------------------
def estimate_layer_flops(block, seqlen=2048):
    flops = 0
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            out, inn = m.weight.shape
            flops += out * inn * seqlen
    print(f"[FLOPS] Layer ~ {flops/1e9:.3f} GFLOPs")
    return flops


# ------------------------------------------------
# 6. Gradient Flow Inspector
# ------------------------------------------------
def inspect_gradients(block, name="block"):
    for n, p in block.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        print(f"[GRAD] {name}.{n} mean={g.mean().item():.6f} std={g.std().item():.6f} max={g.max().item():.6f}")


# ------------------------------------------------
# 7. DSNoT Error Inspection
# ------------------------------------------------
def inspect_dsnot_error(err, name="dsnot_err"):
    print(f"[DSNOT ERR] {name}: mean={err.mean().item():.4f} std={err.std().item():.4f}")
    print(f"[DSNOT ERR] {name}: max={err.max().item():.4f} min={err.min().item():.4f}")


# ------------------------------------------------
# 8. Layer Consistency Check
# ------------------------------------------------
def check_layer(block, name="block"):
    print(f"[CHECK] --- {name} ---")
    for n, m in block.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            if torch.isnan(W).any():
                print(f"[CHECK] {n}: NaN detected!")
            if (W.abs() > 1e3).any():
                print(f"[CHECK] {n}: extreme weight values!")
    print("[CHECK] done")


# ------------------------------------------------
# 9. Dead Neuron Detector
# ------------------------------------------------
def detect_dead_neurons(W_sparse, tol=1e-8):
    row_dead = ((W_sparse.abs().sum(dim=1)) < tol).sum().item()
    col_dead = ((W_sparse.abs().sum(dim=0)) < tol).sum().item()
    print(f"[DEAD] rows={row_dead} cols={col_dead}")


# ------------------------------------------------
# 10. Hook to inspect Qwen2.5 attention projections
# ------------------------------------------------
def inspect_attention(block, x, am=None, pos=None, name="attn"):
    q = block.self_attn.q_proj(x)
    k = block.self_attn.k_proj(x)
    v = block.self_attn.v_proj(x)
    print(f"[ATTN] {name} q mean={q.mean().item():.4f} std={q.std().item():.4f}")
    print(f"[ATTN] {name} k mean={k.mean().item():.4f} std={k.std().item():.4f}")
    print(f"[ATTN] {name} v mean={v.mean().item():.4f} std={v.std().item():.4f}")


# ------------------------------------------------
# 11. Full correctness harness
# ------------------------------------------------
def sanity_check_block(block, xb, am, pos):
    print("[SANITY] Running block forward...")
    with torch.no_grad():
        y = block(xb, attention_mask=am, position_ids=pos)[0]
    inspect_activations(y, "block_out")

    print("[SANITY] Checking gradients…")
    y2 = block(xb, attention_mask=am, position_ids=pos)[0]
    loss = y2.pow(2).mean()
    block.zero_grad()
    loss.backward()
    inspect_gradients(block, "block")


# ------------------------------------------------
# 12. Global system flush
# ------------------------------------------------
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    print("[SYS] GC + CUDA flush done")