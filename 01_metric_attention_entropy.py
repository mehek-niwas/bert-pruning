"""
01_metric_attention_entropy.py  (AE)

Loads the saved base checkpoints, computes per-head attention entropy,
prunes at five ratios × two levels, fine-tunes, evaluates, and
produces publication-ready figures for the final report:

  figures/01_AE_score_heatmap.png          — per-head AE scores (all tasks)
  figures/01_AE_head_accuracy_curves.png   — accuracy vs prune ratio, head level
  figures/01_AE_block_accuracy_curves.png  — accuracy vs prune ratio, block level
  figures/01_AE_head_vs_block.png          — head vs block comparison, per task
  figures/01_AE_pruned_head_map.png        — which heads get pruned at each ratio
  results/01_AE_results.csv               — full results table
  results/01_AE_pruning_report.txt       — full 12×12 head scores + per-layer block means, pruned subsets, eval per ratio
  results/prune_finetune_logs/01_AE/     — JSON log_history + PNG loss/metric curve per (task, head|block, ratio)

Usage:
    python 01_metric_attention_entropy.py
"""

import json
import os, random, copy, csv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_DIR  = "./checkpoints"
FIGURES_DIR     = "./figures"
RESULTS_DIR     = "./results"
for d in [FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

CALIB_SIZE      = 512
PRUNE_RATIOS    = [0.1, 0.2, 0.3, 0.4, 0.5]
FINETUNE_EPOCHS = 2
N_LAYERS        = 12
N_HEADS         = 12

TASK_CONFIG = {
    "sst2": {
        "dataset":       ("glue", "sst2"),
        "text_keys":     ("sentence", None),
        "num_labels":    2,
        "metric_name":   "glue/sst2",
        "primary_key":   "eval_accuracy",
        "primary_label": "Accuracy",
        "color":         "#2563EB",
    },
    "cola": {
        "dataset":       ("glue", "cola"),
        "text_keys":     ("sentence", None),
        "num_labels":    2,
        "metric_name":   "glue/cola",
        "primary_key":   "eval_matthews_correlation",
        "primary_label": "Matthews Corr.",
        "color":         "#16A34A",
    },
    "mrpc": {
        "dataset":       ("glue", "mrpc"),
        "text_keys":     ("sentence1", "sentence2"),
        "num_labels":    2,
        "metric_name":   "glue/mrpc",
        "primary_key":   "eval_f1",
        "primary_label": "F1",
        "color":         "#DC2626",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def make_tokenize_fn(tokenizer, text_keys):
    col1, col2 = text_keys
    def tokenize(batch):
        return tokenizer(batch[col1], truncation=True) if col2 is None \
               else tokenizer(batch[col1], batch[col2], truncation=True)
    return tokenize

def compute_metrics_fn(metric_name):
    metric = evaluate.load(*metric_name.split("/"))
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)
    return compute_metrics

def safe_remove_cols(dataset):
    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    drop = [c for c in dataset.column_names if c not in keep]
    return dataset.remove_columns(drop)

# ---------------------------------------------------------------------------
# Attention entropy computation
# ---------------------------------------------------------------------------
def compute_attention_entropy(model, dataloader):
    """Returns [N_LAYERS, N_HEADS] mean entropy. Higher = more diffuse."""
    model.eval().to(DEVICE)
    entropy_sum      = np.zeros((N_LAYERS, N_HEADS))
    count            = 0
    layer_attentions = {}

    def make_hook(idx):
        def hook(module, inp, out):
            layer_attentions[idx] = out[1].detach().cpu()
        return hook

    hooks = [
        layer.attention.self.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.bert.encoder.layer)
    ]
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model(**batch, output_attentions=True)
            for li, attn in layer_attentions.items():
                ent = -(attn * torch.log(attn + 1e-9)).sum(-1).mean((0, 2))
                entropy_sum[li] += ent.numpy()
            count += 1
    for h in hooks:
        h.remove()
    return entropy_sum / max(count, 1)

# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_heads_by_entropy(model, entropy_scores, prune_ratio):
    """Prune HIGHEST-entropy heads (most diffuse)."""
    model = copy.deepcopy(model)
    n_prune = max(1, int(N_LAYERS * N_HEADS * prune_ratio))
    flat = sorted(
        [(l, h, float(entropy_scores[l, h])) for l in range(N_LAYERS) for h in range(N_HEADS)],
        key=lambda x: x[2], reverse=True,
    )
    heads_to_prune = {}
    for l, h, _ in flat[:n_prune]:
        heads_to_prune.setdefault(l, set()).add(h)
    for l, hset in heads_to_prune.items():
        model.bert.encoder.layer[l].attention.prune_heads(hset)
    return model, flat[:n_prune]


class SkippableBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.skip  = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        if self.skip:
            return (hidden_states,)
        return self.block(hidden_states, attention_mask=attention_mask,
                          head_mask=head_mask, output_attentions=output_attentions)


def prune_blocks_by_entropy(model, entropy_scores, prune_ratio):
    """Skip blocks with HIGHEST mean-entropy across heads."""
    model = copy.deepcopy(model)
    for i, layer in enumerate(model.bert.encoder.layer):
        model.bert.encoder.layer[i] = SkippableBlock(layer)
    n_skip = max(1, int(N_LAYERS * prune_ratio))
    layer_means = entropy_scores.mean(axis=1)
    skip_indices = np.argsort(layer_means)[::-1][:n_skip]
    skip_list = skip_indices.tolist()
    block_details = [(int(i), float(layer_means[i])) for i in skip_list]
    for idx in skip_list:
        model.bert.encoder.layer[idx].skip = True
    return model, skip_list, block_details


AE_HEAD_RULE = (
    "Prune heads with the highest mean attention entropy (most diffuse attention "
    "distributions; AE treats them as least critical and removes them first)."
)
AE_BLOCK_RULE = (
    "Skip entire encoder blocks whose mean attention entropy across heads is highest "
    "(most diffuse layers are skipped first)."
)


def append_complete_score_inventory_ae(lines, task_name, scores):
    """Every head and every layer's block-level mean (before any pruning)."""
    lines.append("")
    lines.append(
        f"COMPLETE SCORE INVENTORY — task={task_name} — "
        "per-head mean attention entropy (all 12×12 heads)"
    )
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            lines.append(
                f"  L{li:2d}  H{hi:2d}  entropy={float(scores[li, hi]):.6f}"
            )
    lines.append(
        "Per-block aggregate: mean entropy over heads in each layer "
        "(block pruning uses these means; highest mean skipped first):"
    )
    layer_means = scores.mean(axis=1)
    for li in range(N_LAYERS):
        lines.append(
            f"  Block L{li:2d}  mean_head_entropy={float(layer_means[li]):.6f}"
        )


def append_pruning_report_head(
    lines, task_name, ratio, pruned_list, res, cfg, baseline_score, primary_key
):
    score = res.get(primary_key, 0.0)
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"Task={task_name}  level=head  prune_ratio={ratio:.0%}  "
        f"n_pruned={len(pruned_list)}"
    )
    lines.append(f"Selection rule (AE): {AE_HEAD_RULE}")
    lines.append("Pruned heads (layer, head, entropy_score):")
    for l, h, s in pruned_list:
        lines.append(f"  L{l:2d}  H{h:2d}  entropy={s:.6f}")
    lines.append(
        f"After fine-tune: {cfg['primary_label']} = {score:.6f}  "
        f"(baseline {baseline_score:.6f},  delta {score - baseline_score:+.6f})"
    )
    ev = ", ".join(
        f"{k}={float(v):.6f}"
        for k, v in sorted(res.items())
        if isinstance(v, (float, np.floating))
    )
    lines.append("Full eval metrics: " + ev)


def append_pruning_report_block(
    lines, task_name, ratio, block_details, res, cfg, baseline_score, primary_key
):
    score = res.get(primary_key, 0.0)
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"Task={task_name}  level=block  prune_ratio={ratio:.0%}  "
        f"n_skipped_blocks={len(block_details)}"
    )
    lines.append(f"Selection rule (AE): {AE_BLOCK_RULE}")
    lines.append("Skipped blocks (encoder layer index, mean entropy over heads):")
    for layer_i, mean_ent in block_details:
        lines.append(f"  Block L{layer_i:2d}  mean_head_entropy={mean_ent:.6f}")
    lines.append(
        f"After fine-tune: {cfg['primary_label']} = {score:.6f}  "
        f"(baseline {baseline_score:.6f},  delta {score - baseline_score:+.6f})"
    )
    ev = ", ".join(
        f"{k}={float(v):.6f}"
        for k, v in sorted(res.items())
        if isinstance(v, (float, np.floating))
    )
    lines.append("Full eval metrics: " + ev)

# ---------------------------------------------------------------------------
# Prune-stage fine-tune: structured logs + loss curves (one JSON + one PNG per run)
# ---------------------------------------------------------------------------
SCRIPT_TAG_AE = "01_AE"


def plot_prune_finetune_curve(log_history, cfg, out_path, title):
    df = pd.DataFrame(log_history)
    color = cfg["color"]
    pk = cfg["primary_key"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx()
    train_df = (
        df[df.get("loss", pd.Series(dtype=float)).notna()].copy()
        if "loss" in df.columns
        else pd.DataFrame()
    )
    eval_df = (
        df[df.get("eval_loss", pd.Series(dtype=float)).notna()].copy()
        if "eval_loss" in df.columns
        else pd.DataFrame()
    )
    if not train_df.empty and "epoch" in train_df.columns:
        ax.plot(
            train_df["epoch"],
            train_df["loss"],
            color=color,
            lw=1.4,
            alpha=0.55,
            label="Train loss",
        )
    if not eval_df.empty and "epoch" in eval_df.columns:
        ax.plot(
            eval_df["epoch"],
            eval_df["eval_loss"],
            color=color,
            lw=2,
            ls="--",
            label="Val loss",
        )
        if pk in eval_df.columns:
            ax2.plot(
                eval_df["epoch"],
                eval_df[pk],
                color="#F59E0B",
                lw=2,
                marker="o",
                ms=5,
                label=cfg["primary_label"],
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color=color)
    ax2.set_ylabel(cfg["primary_label"], color="#F59E0B")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax2.spines["right"].set_visible(True)
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_prune_finetune_logs_and_curve(
    script_tag, metric_title, task_name, level, prune_ratio, trainer, cfg
):
    subdir = os.path.join(RESULTS_DIR, "prune_finetune_logs", script_tag)
    os.makedirs(subdir, exist_ok=True)
    rtag = int(round(prune_ratio * 100))
    base = f"{task_name}_{level}_r{rtag}"
    log_history = trainer.state.log_history
    payload = {
        "script": script_tag,
        "metric": metric_title,
        "task": task_name,
        "level": level,
        "prune_ratio": prune_ratio,
        "primary_key": cfg["primary_key"],
        "log_history": log_history,
    }
    json_path = os.path.join(subdir, f"{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    png_path = os.path.join(subdir, f"{base}.png")
    plot_prune_finetune_curve(
        log_history,
        cfg,
        png_path,
        f"{metric_title} fine-tune — {task_name.upper()} {level} prune {rtag}%",
    )
    print(f"    FT log:  {json_path}")
    print(f"    FT plot: {png_path}")


# ---------------------------------------------------------------------------
# Fine-tune & evaluate
# ---------------------------------------------------------------------------
def finetune_and_eval(
    model,
    task_name,
    cfg,
    tokenized,
    tokenizer,
    *,
    level: str,
    prune_ratio: float,
    script_tag: str = SCRIPT_TAG_AE,
    metric_title: str = "AE",
):
    set_seed(SEED)
    rtag = int(round(prune_ratio * 100))
    out_dir = os.path.join(
        RESULTS_DIR, "prune_finetune_tmp", script_tag, f"{task_name}_{level}_r{rtag}"
    )
    os.makedirs(out_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
        logging_steps=20,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn(cfg["metric_name"]),
    )
    trainer.train()
    save_prune_finetune_logs_and_curve(
        script_tag, metric_title, task_name, level, prune_ratio, trainer, cfg
    )
    return trainer.evaluate()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_ae_score_heatmaps(entropy_per_task):
    """Figure 1 — AE score heatmap per task (which heads are most diffuse)."""
    n = len(entropy_per_task)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    vmin = min(e.min() for e in entropy_per_task.values())
    vmax = max(e.max() for e in entropy_per_task.values())

    for ax, (task_name, entropy) in zip(axes, entropy_per_task.items()):
        sns.heatmap(entropy, ax=ax, cmap="plasma_r", vmin=vmin, vmax=vmax,
                    xticklabels=[f"H{h}" for h in range(N_HEADS)],
                    yticklabels=[f"L{l}" for l in range(N_LAYERS)],
                    cbar=(ax is axes[-1]),
                    linewidths=0.25, linecolor="white",
                    annot=True, fmt=".2f", annot_kws={"size": 6})
        ax.set_title(f"{task_name.upper()}", fontweight="bold")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.tick_params(labelsize=7)

    fig.suptitle("AE Metric: Mean Per-Head Attention Entropy (higher = pruned first)",
                 fontsize=12, y=1.01)
    path = os.path.join(FIGURES_DIR, "01_AE_score_heatmap.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_curves(df, level, metric="AE"):
    """Figure 2/3 — Accuracy vs prune ratio per task."""
    sub = df[df["level"] == level]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    fig.suptitle(f"{metric} Metric — {level.capitalize()}-Level Pruning: "
                 f"Performance vs Prune Ratio", fontsize=13, fontweight="bold")

    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        task_df  = sub[sub["task"] == task_name].sort_values("prune_ratio")
        baseline = task_df["baseline_score"].iloc[0] if not task_df.empty else None
        color    = cfg["color"]

        if not task_df.empty:
            ax.plot(task_df["prune_ratio"] * 100, task_df["primary_score"],
                    color=color, lw=2.5, marker="o", ms=7, label=metric)
            for _, row in task_df.iterrows():
                ax.annotate(f"{row['primary_score']:.3f}",
                            (row["prune_ratio"] * 100, row["primary_score"]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color)
        if baseline is not None:
            ax.axhline(baseline, ls="--", color="gray", lw=1.5, label="Baseline")

        ax.set_title(task_name.upper(), fontweight="bold", color=color)
        ax.set_xlabel("Pruned heads (%)" if level == "head" else "Pruned blocks (%)")
        ax.set_ylabel(cfg["primary_label"])
        ax.set_xticks([r * 100 for r in PRUNE_RATIOS])
        ax.legend(fontsize=9)

    level_tag = "head" if level == "head" else "block"
    path = os.path.join(FIGURES_DIR, f"01_AE_{level_tag}_accuracy_curves.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_head_vs_block(df, metric="AE"):
    """Figure 4 — Head vs block on same axes per task."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{metric} Metric — Head-Level vs Block-Level Pruning",
                 fontsize=13, fontweight="bold")

    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        color = cfg["color"]
        for level, ls, mk in [("head", "-", "o"), ("block", "--", "s")]:
            sub = df[(df["task"] == task_name) & (df["level"] == level)].sort_values("prune_ratio")
            if not sub.empty:
                ax.plot(sub["prune_ratio"] * 100, sub["primary_score"],
                        color=color, lw=2, ls=ls, marker=mk, ms=6,
                        label=level.capitalize())
        baseline = df[df["task"] == task_name]["baseline_score"].iloc[0] \
                   if not df[df["task"] == task_name].empty else None
        if baseline is not None:
            ax.axhline(baseline, ls=":", color="gray", lw=1.5, label="Baseline")
        ax.set_title(task_name.upper(), fontweight="bold", color=color)
        ax.set_xlabel("Prune ratio (%)")
        ax.set_ylabel(cfg["primary_label"])
        ax.set_xticks([r * 100 for r in PRUNE_RATIOS])
        ax.legend(fontsize=9)

    path = os.path.join(FIGURES_DIR, "01_AE_head_vs_block.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_pruned_head_map(pruned_heads_per_task_ratio):
    """Figure 5 — Which heads are pruned at each ratio (head-level)."""
    ratios = PRUNE_RATIOS
    n_tasks = len(TASK_CONFIG)
    fig, axes = plt.subplots(n_tasks, len(ratios),
                             figsize=(4 * len(ratios), 3.5 * n_tasks))
    fig.suptitle("AE: Pruned Heads at Each Ratio (red = pruned)",
                 fontsize=13, fontweight="bold")

    for row_i, task_name in enumerate(TASK_CONFIG.keys()):
        for col_i, ratio in enumerate(ratios):
            ax  = axes[row_i][col_i]
            grid = np.zeros((N_LAYERS, N_HEADS))
            pruned = pruned_heads_per_task_ratio.get((task_name, ratio), [])
            for l, h, _ in pruned:
                grid[l, h] = 1.0
            sns.heatmap(grid, ax=ax, cmap=["#E2E8F0", "#DC2626"],
                        vmin=0, vmax=1, cbar=False,
                        xticklabels=[str(h) for h in range(N_HEADS)],
                        yticklabels=[str(l) for l in range(N_LAYERS)],
                        linewidths=0.3, linecolor="white")
            ax.set_title(f"{task_name.upper()} | {int(ratio*100)}%",
                         fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)

    path = os.path.join(FIGURES_DIR, "01_AE_pruned_head_map.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution(entropy_per_task):
    """Figure 6 — Distribution of AE scores across all heads per task."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for task_name, entropy in entropy_per_task.items():
        scores = entropy.flatten()
        ax.hist(scores, bins=20, alpha=0.55, density=True,
                color=TASK_CONFIG[task_name]["color"],
                edgecolor="white", label=task_name.upper())
        ax.axvline(scores.mean(), color=TASK_CONFIG[task_name]["color"],
                   lw=2, ls="--")

    ax.set_xlabel("Attention Entropy Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of AE Scores Across All Heads",
                 fontweight="bold", fontsize=13)
    ax.legend()
    path = os.path.join(FIGURES_DIR, "01_AE_score_distribution.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows                         = []
    entropy_per_task             = {}
    pruned_heads_per_task_ratio  = {}  # (task, ratio) -> list of (l, h, score)
    report_lines                 = [
        "01_metric_attention_entropy.py — AE pruning report",
        "Metric: mean per-head attention entropy on calibration data.",
        "",
    ]

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"  AE — Task: {task_name.upper()}")
        print(f"{'='*60}")

        ckpt_path  = os.path.join(CHECKPOINT_DIR, f"bert-{task_name}", "best")
        tokenizer  = AutoTokenizer.from_pretrained(ckpt_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)

        raw       = load_dataset(*cfg["dataset"])
        tokenized = raw.map(make_tokenize_fn(tokenizer, cfg["text_keys"]), batched=True)
        collator  = DataCollatorWithPadding(tokenizer)

        calib_loader = torch.utils.data.DataLoader(
            safe_remove_cols(tokenized["train"].select(range(CALIB_SIZE))),
            batch_size=32, collate_fn=collator,
        )

        print("  Computing attention entropy...")
        entropy = compute_attention_entropy(base_model, calib_loader)
        entropy_per_task[task_name] = entropy
        print(f"  Entropy — min: {entropy.min():.3f}  max: {entropy.max():.3f}")

        # Baseline score (no pruning)
        baseline_eval = Trainer(
            model=copy.deepcopy(base_model),
            args=TrainingArguments(
                output_dir="./tmp_baseline_AE", report_to="none",
                per_device_eval_batch_size=64, seed=SEED,
            ),
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics_fn(cfg["metric_name"]),
        ).evaluate()
        baseline_score = baseline_eval.get(cfg["primary_key"], 0.0)
        pk = cfg["primary_key"]
        report_lines.append("")
        report_lines.append("=" * 72)
        report_lines.append(f"TASK {task_name.upper()} — baseline (no pruning)")
        report_lines.append(
            f"  {cfg['primary_label']} = {baseline_score:.6f}  ({pk})"
        )
        append_complete_score_inventory_ae(report_lines, task_name, entropy)

        # Head-level
        for ratio in PRUNE_RATIOS:
            print(f"\n  [Head] prune_ratio={ratio}")
            print(f"    {AE_HEAD_RULE}")
            pruned, pruned_list = prune_heads_by_entropy(base_model, entropy, ratio)
            pruned_heads_per_task_ratio[(task_name, ratio)] = pruned_list
            res = finetune_and_eval(
                pruned,
                task_name,
                cfg,
                tokenized,
                tokenizer,
                level="head",
                prune_ratio=ratio,
            )
            score = res.get(pk, 0.0)
            print(f"    {cfg['primary_label']}: {score:.4f}  (baseline: {baseline_score:.4f})")
            print(f"    Pruned {len(pruned_list)} heads; listing in report file.")
            append_pruning_report_head(
                report_lines, task_name, ratio, pruned_list, res, cfg, baseline_score, pk
            )
            rows.append(dict(task=task_name, metric="AE", level="head",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

        # Block-level
        for ratio in PRUNE_RATIOS:
            print(f"\n  [Block] prune_ratio={ratio}")
            print(f"    {AE_BLOCK_RULE}")
            pruned, _, block_details = prune_blocks_by_entropy(
                base_model, entropy, ratio
            )
            for layer_i, mean_ent in block_details:
                print(f"    Skip block L{layer_i}: mean head entropy = {mean_ent:.6f}")
            res = finetune_and_eval(
                pruned,
                task_name,
                cfg,
                tokenized,
                tokenizer,
                level="block",
                prune_ratio=ratio,
            )
            score = res.get(pk, 0.0)
            print(f"    {cfg['primary_label']}: {score:.4f}")
            append_pruning_report_block(
                report_lines, task_name, ratio, block_details, res, cfg, baseline_score, pk
            )
            rows.append(dict(task=task_name, metric="AE", level="block",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

    # ---- Save CSV ----
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "01_AE_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    report_path = os.path.join(RESULTS_DIR, "01_AE_pruning_report.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines) + "\n")
    print(f"Pruning report saved: {report_path}")

    # ---- Print summary table ----
    print("\n" + "="*70)
    print("SUMMARY TABLE — AE Metric")
    print("="*70)
    summary = df[["task", "level", "prune_ratio", "primary_score",
                  "baseline_score", "score_drop"]].copy()
    summary["prune_ratio"] = (summary["prune_ratio"] * 100).astype(int).astype(str) + "%"
    summary["primary_score"]   = summary["primary_score"].round(4)
    summary["baseline_score"]  = summary["baseline_score"].round(4)
    summary["score_drop"]      = summary["score_drop"].round(4)
    print(summary.to_string(index=False))

    # ---- Figures ----
    print("\nGenerating figures...")
    plot_ae_score_heatmaps(entropy_per_task)
    plot_accuracy_curves(df, level="head")
    plot_accuracy_curves(df, level="block")
    plot_head_vs_block(df)
    plot_pruned_head_map(pruned_heads_per_task_ratio)
    plot_score_distribution(entropy_per_task)

    print("\nFigures produced:")
    for name in [
        "01_AE_score_heatmap.png",
        "01_AE_head_accuracy_curves.png",
        "01_AE_block_accuracy_curves.png",
        "01_AE_head_vs_block.png",
        "01_AE_pruned_head_map.png",
        "01_AE_score_distribution.png",
    ]:
        print(f"  {name}")

if __name__ == "__main__":
    main()
