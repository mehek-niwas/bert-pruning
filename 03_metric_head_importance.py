"""
03_metric_head_importance.py  (HIS)

Loads the saved base checkpoints, computes Head Importance Scores via
gradient × head-mask, prunes at five ratios × two levels, fine-tunes,
evaluates, and produces publication-ready figures for the final report:

  figures/03_HIS_score_heatmap.png         — per-head HIS scores
  figures/03_HIS_head_accuracy_curves.png  — accuracy vs prune ratio, head level
  figures/03_HIS_block_accuracy_curves.png — accuracy vs prune ratio, block level
  figures/03_HIS_head_vs_block.png         — head vs block comparison
  figures/03_HIS_pruned_head_map.png       — which heads pruned at each ratio
  figures/03_HIS_score_distribution.png    — distribution of HIS scores
  figures/03_ALL_metric_comparison.png     — AE vs KL-R vs HIS on same axes
  results/03_HIS_results.csv
  results/ALL_metrics_combined.csv         — merged table for all three metrics
  results/03_HIS_pruning_report.txt      — full 12×12 head scores + per-layer block means, pruned subsets, eval per ratio
  results/prune_finetune_logs/03_HIS/    — JSON log_history + PNG loss/metric curve per (task, head|block, ratio)

Usage:
    python 03_metric_head_importance.py
"""

import json
import os, random, copy
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
FINETUNE_EPOCHS = 3
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

# Colours and markers for the cross-metric comparison plot
METRIC_STYLE = {
    "AE":   {"color": "#7C3AED", "marker": "o", "ls": "-"},
    "KL-R": {"color": "#D97706", "marker": "D", "ls": "--"},
    "HIS":  {"color": "#0F172A", "marker": "s", "ls": "-."},
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
    return dataset.remove_columns([c for c in dataset.column_names if c not in keep])

# ---------------------------------------------------------------------------
# HIS computation
# ---------------------------------------------------------------------------
def compute_head_importance(model, dataloader):
    """
    Returns [N_LAYERS, N_HEADS] mean HIS.
    HIS(l, h) = mean | grad(loss w.r.t. head_mask[l,h]) * mask[l,h] |
    Requires a full forward+backward per batch (mask initialised to ones).
    Lower HIS -> less effect on loss -> prune first.
    """
    model.train().to(DEVICE)       # needs grads
    importance_sum = np.zeros((N_LAYERS, N_HEADS))
    n_batches      = 0

    for batch in dataloader:
        head_mask = torch.ones(N_LAYERS, N_HEADS, requires_grad=True, device=DEVICE)

        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            head_mask=head_mask,
        )
        outputs.loss.backward()

        with torch.no_grad():
            importance_sum += (head_mask.grad * head_mask).abs().cpu().numpy()
        n_batches += 1

    model.eval()
    return importance_sum / max(n_batches, 1)

# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_heads_by_his(model, his_scores, prune_ratio):
    """Prune LOWEST-HIS heads (least important)."""
    model = copy.deepcopy(model)
    n_prune = max(1, int(N_LAYERS * N_HEADS * prune_ratio))
    flat = sorted(
        [(l, h, float(his_scores[l, h])) for l in range(N_LAYERS) for h in range(N_HEADS)],
        key=lambda x: x[2],   # ascending
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


def prune_blocks_by_his(model, his_scores, prune_ratio):
    """Skip blocks with LOWEST mean HIS."""
    model = copy.deepcopy(model)
    for i, layer in enumerate(model.bert.encoder.layer):
        model.bert.encoder.layer[i] = SkippableBlock(layer)
    n_skip = max(1, int(N_LAYERS * prune_ratio))
    layer_means = his_scores.mean(axis=1)
    skip_indices = np.argsort(layer_means)[:n_skip]
    skip_list = skip_indices.tolist()
    block_details = [(int(i), float(layer_means[i])) for i in skip_list]
    for idx in skip_list:
        model.bert.encoder.layer[idx].skip = True
    return model, skip_list, block_details


HIS_HEAD_RULE = (
    "Prune heads with the lowest Head Importance Score: mean |∂loss/∂head_mask · mask| "
    "on calibration batches (smallest gradient signal = least important for the loss)."
)
HIS_BLOCK_RULE = (
    "Skip encoder blocks with the lowest mean HIS across heads "
    "(layers least important to the loss on average are skipped first)."
)


def append_complete_score_inventory_his(lines, task_name, scores):
    """Every head and every layer's block-level mean (before any pruning)."""
    lines.append("")
    lines.append(
        f"COMPLETE SCORE INVENTORY — task={task_name} — "
        "per-head HIS (all 12×12 heads)"
    )
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            lines.append(
                f"  L{li:2d}  H{hi:2d}  HIS={float(scores[li, hi]):.6f}"
            )
    lines.append(
        "Per-block aggregate: mean HIS over heads in each layer "
        "(block pruning uses these means; lowest mean skipped first):"
    )
    layer_means = scores.mean(axis=1)
    for li in range(N_LAYERS):
        lines.append(
            f"  Block L{li:2d}  mean_head_HIS={float(layer_means[li]):.6f}"
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
    lines.append(f"Selection rule (HIS): {HIS_HEAD_RULE}")
    lines.append("Pruned heads (layer, head, HIS score):")
    for l, h, s in pruned_list:
        lines.append(f"  L{l:2d}  H{h:2d}  HIS={s:.6f}")
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
    lines.append(f"Selection rule (HIS): {HIS_BLOCK_RULE}")
    lines.append("Skipped blocks (layer index, mean HIS over heads):")
    for layer_i, mean_his in block_details:
        lines.append(f"  Block L{layer_i:2d}  mean_head_HIS={mean_his:.6f}")
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
SCRIPT_TAG_HIS = "03_HIS"


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
    script_tag: str = SCRIPT_TAG_HIS,
    metric_title: str = "HIS",
):
    set_seed(SEED)
    rtag = int(round(prune_ratio * 100))
    out_dir = os.path.join(
        RESULTS_DIR, "prune_finetune_tmp", script_tag, f"{task_name}_{level}_r{rtag}"
    )
    os.makedirs(out_dir, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=FINETUNE_EPOCHS,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="no",
            seed=SEED,
            data_seed=SEED,
            report_to="none",
            fp16=torch.cuda.is_available(),
            logging_steps=20,
        ),
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
# HIS-specific figures
# ---------------------------------------------------------------------------
def plot_his_score_heatmaps(his_per_task):
    n = len(his_per_task)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    vmin = min(e.min() for e in his_per_task.values())
    vmax = max(e.max() for e in his_per_task.values())

    for ax, (task_name, his) in zip(axes, his_per_task.items()):
        sns.heatmap(his, ax=ax, cmap="magma_r", vmin=vmin, vmax=vmax,
                    xticklabels=[f"H{h}" for h in range(N_HEADS)],
                    yticklabels=[f"L{l}" for l in range(N_LAYERS)],
                    cbar=(ax is axes[-1]),
                    linewidths=0.25, linecolor="white",
                    annot=True, fmt=".3f", annot_kws={"size": 5.5})
        ax.set_title(f"{task_name.upper()}", fontweight="bold")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.tick_params(labelsize=7)

    fig.suptitle("HIS Metric: Head Importance Score  (lower = pruned first)",
                 fontsize=12, y=1.01)
    path = os.path.join(FIGURES_DIR, "03_HIS_score_heatmap.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_curves(df, level):
    sub = df[df["level"] == level]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"HIS Metric — {level.capitalize()}-Level Pruning: "
                 "Performance vs Prune Ratio", fontsize=13, fontweight="bold")
    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        task_df  = sub[sub["task"] == task_name].sort_values("prune_ratio")
        baseline = task_df["baseline_score"].iloc[0] if not task_df.empty else None
        color    = cfg["color"]
        if not task_df.empty:
            ax.plot(task_df["prune_ratio"] * 100, task_df["primary_score"],
                    color=color, lw=2.5, marker="s", ms=7, label="HIS")
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
    path = os.path.join(FIGURES_DIR, f"03_HIS_{level}_accuracy_curves.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_head_vs_block(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("HIS Metric — Head-Level vs Block-Level Pruning",
                 fontsize=13, fontweight="bold")
    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        color = cfg["color"]
        for level, ls, mk in [("head", "-", "s"), ("block", "--", "^")]:
            sub = df[(df["task"] == task_name) & (df["level"] == level)].sort_values("prune_ratio")
            if not sub.empty:
                ax.plot(sub["prune_ratio"] * 100, sub["primary_score"],
                        color=color, lw=2, ls=ls, marker=mk, ms=6, label=level.capitalize())
        bdf = df[df["task"] == task_name]
        if not bdf.empty:
            ax.axhline(bdf["baseline_score"].iloc[0], ls=":", color="gray", lw=1.5, label="Baseline")
        ax.set_title(task_name.upper(), fontweight="bold", color=color)
        ax.set_xlabel("Prune ratio (%)"); ax.set_ylabel(cfg["primary_label"])
        ax.set_xticks([r * 100 for r in PRUNE_RATIOS]); ax.legend(fontsize=9)
    path = os.path.join(FIGURES_DIR, "03_HIS_head_vs_block.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_pruned_head_map(pruned_per_task_ratio):
    ratios  = PRUNE_RATIOS
    n_tasks = len(TASK_CONFIG)
    fig, axes = plt.subplots(n_tasks, len(ratios),
                             figsize=(4 * len(ratios), 3.5 * n_tasks))
    fig.suptitle("HIS: Pruned Heads at Each Ratio (red = pruned)",
                 fontsize=13, fontweight="bold")
    for row_i, task_name in enumerate(TASK_CONFIG.keys()):
        for col_i, ratio in enumerate(ratios):
            ax   = axes[row_i][col_i]
            grid = np.zeros((N_LAYERS, N_HEADS))
            for l, h, _ in pruned_per_task_ratio.get((task_name, ratio), []):
                grid[l, h] = 1.0
            sns.heatmap(grid, ax=ax, cmap=["#E2E8F0", "#0F172A"],
                        vmin=0, vmax=1, cbar=False,
                        xticklabels=list(range(N_HEADS)),
                        yticklabels=list(range(N_LAYERS)),
                        linewidths=0.3, linecolor="white")
            ax.set_title(f"{task_name.upper()} | {int(ratio*100)}%",
                         fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)
    path = os.path.join(FIGURES_DIR, "03_HIS_pruned_head_map.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution(his_per_task):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for task_name, his in his_per_task.items():
        scores = his.flatten()
        ax.hist(scores, bins=20, alpha=0.55, density=True,
                color=TASK_CONFIG[task_name]["color"],
                edgecolor="white", label=task_name.upper())
        ax.axvline(scores.mean(), color=TASK_CONFIG[task_name]["color"], lw=2, ls="--")
    ax.set_xlabel("HIS Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of HIS Scores Across All Heads",
                 fontweight="bold", fontsize=13)
    ax.legend()
    path = os.path.join(FIGURES_DIR, "03_HIS_score_distribution.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Cross-metric comparison  (requires prior CSVs from scripts 01 and 02)
# ---------------------------------------------------------------------------
def plot_all_metrics_comparison(his_df):
    ae_path  = os.path.join(RESULTS_DIR, "01_AE_results.csv")
    klr_path = os.path.join(RESULTS_DIR, "02_KLR_results.csv")

    dfs = [his_df]
    if os.path.exists(ae_path):
        dfs.append(pd.read_csv(ae_path))
    else:
        print("  Warning: 01_AE_results.csv not found — skipping AE in comparison plot.")
    if os.path.exists(klr_path):
        dfs.append(pd.read_csv(klr_path))
    else:
        print("  Warning: 02_KLR_results.csv not found — skipping KL-R in comparison plot.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "ALL_metrics_combined.csv"), index=False)
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'ALL_metrics_combined.csv')}")

    for level in ["head", "block"]:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(f"All Metrics — {level.capitalize()}-Level Pruning Comparison",
                     fontsize=14, fontweight="bold")

        for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
            sub = combined[(combined["task"] == task_name) & (combined["level"] == level)]
            baseline = sub["baseline_score"].iloc[0] if not sub.empty else None

            for metric_name, style in METRIC_STYLE.items():
                mdf = sub[sub["metric"] == metric_name].sort_values("prune_ratio")
                if not mdf.empty:
                    ax.plot(mdf["prune_ratio"] * 100, mdf["primary_score"],
                            color=style["color"], lw=2.2,
                            ls=style["ls"], marker=style["marker"],
                            ms=7, label=metric_name)

            if baseline is not None:
                ax.axhline(baseline, ls=":", color="#94A3B8", lw=1.5, label="Baseline")

            ax.set_title(task_name.upper(), fontweight="bold",
                         color=cfg["color"], fontsize=12)
            ax.set_xlabel("Prune ratio (%)")
            ax.set_ylabel(cfg["primary_label"])
            ax.set_xticks([r * 100 for r in PRUNE_RATIOS])
            ax.legend(fontsize=9)

        path = os.path.join(FIGURES_DIR, f"03_ALL_metric_comparison_{level}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        print(f"  Saved: {path}")

    # Score-drop bar chart at 30% and 50% (clearest comparison points)
    for ratio in [0.3, 0.5]:
        for level in ["head", "block"]:
            sub   = combined[(combined["prune_ratio"] == ratio) & (combined["level"] == level)]
            tasks = list(TASK_CONFIG.keys())
            metrics = list(METRIC_STYLE.keys())

            fig, ax = plt.subplots(figsize=(9, 4.5))
            x      = np.arange(len(tasks))
            width  = 0.25
            for i, metric_name in enumerate(metrics):
                mdf    = sub[sub["metric"] == metric_name]
                scores = [mdf[mdf["task"] == t]["score_drop"].values[0]
                          if not mdf[mdf["task"] == t].empty else 0.0 for t in tasks]
                bars = ax.bar(x + i * width, scores,
                              width, label=metric_name,
                              color=METRIC_STYLE[metric_name]["color"],
                              edgecolor="white", alpha=0.85)
                for bar, s in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{s:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x + width)
            ax.set_xticklabels([t.upper() for t in tasks], fontsize=11)
            ax.set_ylabel("Score drop vs baseline")
            ax.set_title(
                f"Score Drop at {int(ratio*100)}% Pruning — "
                f"{level.capitalize()}-Level",
                fontweight="bold", fontsize=13
            )
            ax.legend()
            ax.axhline(0, color="black", lw=0.8)
            path = os.path.join(FIGURES_DIR,
                f"03_ALL_score_drop_{level}_{int(ratio*100)}pct.png")
            plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
            print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows                  = []
    his_per_task          = {}
    pruned_per_task_ratio = {}
    report_lines          = [
        "03_metric_head_importance.py — HIS pruning report",
        "Metric: mean |gradient of loss w.r.t. head_mask × mask| on calibration data.",
        "",
    ]

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"  HIS — Task: {task_name.upper()}")
        print(f"{'='*60}")

        ckpt_path  = os.path.join(CHECKPOINT_DIR, f"bert-{task_name}", "best")
        tokenizer  = AutoTokenizer.from_pretrained(ckpt_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        raw        = load_dataset(*cfg["dataset"])
        tokenized  = raw.map(make_tokenize_fn(tokenizer, cfg["text_keys"]), batched=True)
        collator   = DataCollatorWithPadding(tokenizer)

        calib_loader = torch.utils.data.DataLoader(
            safe_remove_cols(tokenized["train"].select(range(CALIB_SIZE))),
            batch_size=32, collate_fn=collator,
        )

        print("  Computing Head Importance Scores (requires gradients)...")
        his = compute_head_importance(base_model, calib_loader)
        his_per_task[task_name] = his
        print(f"  HIS — min: {his.min():.6f}  max: {his.max():.6f}")

        # Baseline
        baseline_eval = Trainer(
            model=copy.deepcopy(base_model),
            args=TrainingArguments(output_dir="./tmp_baseline_HIS",
                                   report_to="none", per_device_eval_batch_size=64, seed=SEED),
            eval_dataset=tokenized["validation"], tokenizer=tokenizer,
            data_collator=collator, compute_metrics=compute_metrics_fn(cfg["metric_name"]),
        ).evaluate()
        baseline_score = baseline_eval.get(cfg["primary_key"], 0.0)
        pk = cfg["primary_key"]
        report_lines.append("")
        report_lines.append("=" * 72)
        report_lines.append(f"TASK {task_name.upper()} — baseline (no pruning)")
        report_lines.append(
            f"  {cfg['primary_label']} = {baseline_score:.6f}  ({pk})"
        )
        append_complete_score_inventory_his(report_lines, task_name, his)

        for ratio in PRUNE_RATIOS:
            print(f"\n  [Head] prune_ratio={ratio}")
            print(f"    {HIS_HEAD_RULE}")
            pruned, pruned_list = prune_heads_by_his(base_model, his, ratio)
            pruned_per_task_ratio[(task_name, ratio)] = pruned_list
            res   = finetune_and_eval(
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
            print(f"    Pruned {len(pruned_list)} heads; full list in report file.")
            append_pruning_report_head(
                report_lines, task_name, ratio, pruned_list, res, cfg, baseline_score, pk
            )
            rows.append(dict(task=task_name, metric="HIS", level="head",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

        for ratio in PRUNE_RATIOS:
            print(f"\n  [Block] prune_ratio={ratio}")
            print(f"    {HIS_BLOCK_RULE}")
            pruned, _, block_details = prune_blocks_by_his(base_model, his, ratio)
            for layer_i, mean_his in block_details:
                print(f"    Skip block L{layer_i}: mean head HIS = {mean_his:.6f}")
            res   = finetune_and_eval(
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
            rows.append(dict(task=task_name, metric="HIS", level="block",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "03_HIS_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    report_path = os.path.join(RESULTS_DIR, "03_HIS_pruning_report.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines) + "\n")
    print(f"Pruning report saved: {report_path}")

    print("\n" + "="*70)
    print("SUMMARY TABLE — HIS Metric")
    print("="*70)
    summary = df[["task", "level", "prune_ratio", "primary_score",
                  "baseline_score", "score_drop"]].copy()
    summary["prune_ratio"]    = (summary["prune_ratio"] * 100).astype(int).astype(str) + "%"
    summary["primary_score"]  = summary["primary_score"].round(4)
    summary["baseline_score"] = summary["baseline_score"].round(4)
    summary["score_drop"]     = summary["score_drop"].round(4)
    print(summary.to_string(index=False))

    print("\nGenerating figures...")
    plot_his_score_heatmaps(his_per_task)
    plot_accuracy_curves(df, "head")
    plot_accuracy_curves(df, "block")
    plot_head_vs_block(df)
    plot_pruned_head_map(pruned_per_task_ratio)
    plot_score_distribution(his_per_task)
    plot_all_metrics_comparison(df)   # merges all three CSVs

    print("\nFigures produced:")
    for name in [
        "03_HIS_score_heatmap.png",
        "03_HIS_head_accuracy_curves.png",
        "03_HIS_block_accuracy_curves.png",
        "03_HIS_head_vs_block.png",
        "03_HIS_pruned_head_map.png",
        "03_HIS_score_distribution.png",
        "03_ALL_metric_comparison_head.png",
        "03_ALL_metric_comparison_block.png",
        "03_ALL_score_drop_head_30pct.png",
        "03_ALL_score_drop_head_50pct.png",
        "03_ALL_score_drop_block_30pct.png",
        "03_ALL_score_drop_block_50pct.png",
    ]:
        print(f"  {name}")
    print("\nAll done. Run scripts 01 and 02 before this one for the full comparison plots.")

if __name__ == "__main__":
    main()
