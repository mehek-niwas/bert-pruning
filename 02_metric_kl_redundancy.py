"""
02_metric_kl_redundancy.py  (KL-R)

Loads the saved base checkpoints, computes pairwise KL redundancy scores,
prunes at five ratios × two levels, fine-tunes, evaluates, and produces
publication-ready figures for the final report:

  figures/02_KLR_score_heatmap.png         — per-head KL-R scores
  figures/02_KLR_head_accuracy_curves.png  — accuracy vs prune ratio, head level
  figures/02_KLR_block_accuracy_curves.png — accuracy vs prune ratio, block level
  figures/02_KLR_head_vs_block.png         — head vs block comparison
  figures/02_KLR_pruned_head_map.png       — which heads pruned at each ratio
  figures/02_KLR_score_distribution.png    — distribution of KL-R scores
  results/02_KLR_results.csv

Usage:
    python 02_metric_kl_redundancy.py
"""

import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# KL Redundancy computation
# ---------------------------------------------------------------------------
def compute_kl_redundancy(model, dataloader):
    """
    Returns [N_LAYERS, N_HEADS] array.
    kl_scores[l, h] = mean over calibration batches of
        min_{k != h} KL(attn_h || attn_k)
    Lower = more redundant = prune first.
    """
    model.eval().to(DEVICE)
    kl_sum           = np.zeros((N_LAYERS, N_HEADS))
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
                B, H, S, _ = attn.shape
                layer_kl = np.zeros(H)
                for h in range(H):
                    p = attn[:, h]
                    min_kl = float("inf")
                    for k in range(H):
                        if k == h:
                            continue
                        q  = attn[:, k]
                        kl = F.kl_div(torch.log(q + 1e-9), p,
                                       reduction="none").sum(-1).mean().item()
                        if kl < min_kl:
                            min_kl = kl
                    layer_kl[h] = min_kl
                kl_sum[li] += layer_kl
            count += 1

    for h in hooks:
        h.remove()
    return kl_sum / max(count, 1)

# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_heads_by_kl(model, kl_scores, prune_ratio):
    """Prune LOWEST-KL heads (most redundant)."""
    model = copy.deepcopy(model)
    n_prune = max(1, int(N_LAYERS * N_HEADS * prune_ratio))
    flat = sorted(
        [(l, h, float(kl_scores[l, h])) for l in range(N_LAYERS) for h in range(N_HEADS)],
        key=lambda x: x[2],  # ascending: lowest = most redundant
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


def prune_blocks_by_kl(model, kl_scores, prune_ratio):
    """Skip blocks with LOWEST mean KL (most redundant)."""
    model = copy.deepcopy(model)
    for i, layer in enumerate(model.bert.encoder.layer):
        model.bert.encoder.layer[i] = SkippableBlock(layer)
    n_skip       = max(1, int(N_LAYERS * prune_ratio))
    skip_indices = np.argsort(kl_scores.mean(axis=1))[:n_skip]
    for idx in skip_indices:
        model.bert.encoder.layer[idx].skip = True
    return model, skip_indices.tolist()

# ---------------------------------------------------------------------------
# Fine-tune & evaluate
# ---------------------------------------------------------------------------
def finetune_and_eval(model, task_name, cfg, tokenized, tokenizer):
    set_seed(SEED)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./tmp_{task_name}_KLR",
            num_train_epochs=FINETUNE_EPOCHS,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
            evaluation_strategy="epoch", save_strategy="no",
            seed=SEED, data_seed=SEED, report_to="none",
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn(cfg["metric_name"]),
    )
    trainer.train()
    return trainer.evaluate()

# ---------------------------------------------------------------------------
# Figures  (reuse same plotting pattern as AE script, labelled KLR)
# ---------------------------------------------------------------------------
def plot_klr_score_heatmaps(kl_per_task):
    n = len(kl_per_task)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    vmin = min(e.min() for e in kl_per_task.values())
    vmax = max(e.max() for e in kl_per_task.values())

    for ax, (task_name, kl) in zip(axes, kl_per_task.items()):
        sns.heatmap(kl, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                    xticklabels=[f"H{h}" for h in range(N_HEADS)],
                    yticklabels=[f"L{l}" for l in range(N_LAYERS)],
                    cbar=(ax is axes[-1]),
                    linewidths=0.25, linecolor="white",
                    annot=True, fmt=".2f", annot_kws={"size": 6})
        ax.set_title(f"{task_name.upper()}", fontweight="bold")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.tick_params(labelsize=7)

    fig.suptitle("KL-R Metric: Min Pairwise KL per Head  "
                 "(lower = more redundant = pruned first)", fontsize=12, y=1.01)
    path = os.path.join(FIGURES_DIR, "02_KLR_score_heatmap.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_curves(df, level, metric="KLR"):
    sub = df[df["level"] == level]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"KL-R Metric — {level.capitalize()}-Level Pruning: "
                 f"Performance vs Prune Ratio", fontsize=13, fontweight="bold")

    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        task_df  = sub[sub["task"] == task_name].sort_values("prune_ratio")
        baseline = task_df["baseline_score"].iloc[0] if not task_df.empty else None
        color    = cfg["color"]

        if not task_df.empty:
            ax.plot(task_df["prune_ratio"] * 100, task_df["primary_score"],
                    color=color, lw=2.5, marker="D", ms=7, label="KL-R")
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

    path = os.path.join(FIGURES_DIR, f"02_KLR_{level}_accuracy_curves.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_head_vs_block(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("KL-R Metric — Head-Level vs Block-Level Pruning",
                 fontsize=13, fontweight="bold")
    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        color = cfg["color"]
        for level, ls, mk in [("head", "-", "D"), ("block", "--", "s")]:
            sub = df[(df["task"] == task_name) & (df["level"] == level)].sort_values("prune_ratio")
            if not sub.empty:
                ax.plot(sub["prune_ratio"] * 100, sub["primary_score"],
                        color=color, lw=2, ls=ls, marker=mk, ms=6, label=level.capitalize())
        bdf = df[df["task"] == task_name]
        if not bdf.empty:
            ax.axhline(bdf["baseline_score"].iloc[0], ls=":", color="gray", lw=1.5, label="Baseline")
        ax.set_title(task_name.upper(), fontweight="bold", color=color)
        ax.set_xlabel("Prune ratio (%)"); ax.set_ylabel(cfg["primary_label"])
        ax.set_xticks([r * 100 for r in PRUNE_RATIOS])
        ax.legend(fontsize=9)

    path = os.path.join(FIGURES_DIR, "02_KLR_head_vs_block.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_pruned_head_map(pruned_per_task_ratio):
    ratios  = PRUNE_RATIOS
    n_tasks = len(TASK_CONFIG)
    fig, axes = plt.subplots(n_tasks, len(ratios),
                             figsize=(4 * len(ratios), 3.5 * n_tasks))
    fig.suptitle("KL-R: Pruned Heads at Each Ratio (red = pruned)",
                 fontsize=13, fontweight="bold")
    for row_i, task_name in enumerate(TASK_CONFIG.keys()):
        for col_i, ratio in enumerate(ratios):
            ax   = axes[row_i][col_i]
            grid = np.zeros((N_LAYERS, N_HEADS))
            for l, h, _ in pruned_per_task_ratio.get((task_name, ratio), []):
                grid[l, h] = 1.0
            sns.heatmap(grid, ax=ax, cmap=["#E2E8F0", "#16A34A"],
                        vmin=0, vmax=1, cbar=False,
                        xticklabels=list(range(N_HEADS)),
                        yticklabels=list(range(N_LAYERS)),
                        linewidths=0.3, linecolor="white")
            ax.set_title(f"{task_name.upper()} | {int(ratio*100)}%",
                         fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)

    path = os.path.join(FIGURES_DIR, "02_KLR_pruned_head_map.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution(kl_per_task):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for task_name, kl in kl_per_task.items():
        scores = kl.flatten()
        ax.hist(scores, bins=20, alpha=0.55, density=True,
                color=TASK_CONFIG[task_name]["color"],
                edgecolor="white", label=task_name.upper())
        ax.axvline(scores.mean(), color=TASK_CONFIG[task_name]["color"], lw=2, ls="--")
    ax.set_xlabel("Min Pairwise KL Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of KL-R Scores Across All Heads",
                 fontweight="bold", fontsize=13)
    ax.legend()
    path = os.path.join(FIGURES_DIR, "02_KLR_score_distribution.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_kl_matrix_heatmap(kl_per_task):
    """Bonus: mean pairwise KL matrix within a representative layer."""
    # We can't recover the full pairwise matrix post-hoc from per-head min-KL,
    # so we plot the per-head KL scores as a vector across layers instead.
    fig, axes = plt.subplots(1, len(kl_per_task), figsize=(5 * len(kl_per_task), 4))
    if len(kl_per_task) == 1:
        axes = [axes]
    for ax, (task_name, kl) in zip(axes, kl_per_task.items()):
        mean_per_layer = kl.mean(axis=1)
        ax.barh(range(N_LAYERS), mean_per_layer,
                color=TASK_CONFIG[task_name]["color"], alpha=0.8)
        ax.set_title(task_name.upper(), fontweight="bold",
                     color=TASK_CONFIG[task_name]["color"])
        ax.set_xlabel("Mean KL-R Score")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
        ax.invert_yaxis()
    fig.suptitle("KL-R: Mean Redundancy Score per Layer  (lower = more redundant layer)",
                 fontsize=12, y=1.01)
    path = os.path.join(FIGURES_DIR, "02_KLR_mean_per_layer.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows                    = []
    kl_per_task             = {}
    pruned_per_task_ratio   = {}

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"  KL-R — Task: {task_name.upper()}")
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

        print("  Computing pairwise KL redundancy (O(H²) per layer — may take a few minutes)...")
        kl = compute_kl_redundancy(base_model, calib_loader)
        kl_per_task[task_name] = kl
        print(f"  KL scores — min: {kl.min():.4f}  max: {kl.max():.4f}")

        # Baseline
        baseline_eval = Trainer(
            model=copy.deepcopy(base_model),
            args=TrainingArguments(output_dir="./tmp_baseline_KLR",
                                   report_to="none", per_device_eval_batch_size=64, seed=SEED),
            eval_dataset=tokenized["validation"], tokenizer=tokenizer,
            data_collator=collator, compute_metrics=compute_metrics_fn(cfg["metric_name"]),
        ).evaluate()
        baseline_score = baseline_eval.get(cfg["primary_key"], 0.0)

        for ratio in PRUNE_RATIOS:
            print(f"\n  [Head] prune_ratio={ratio}")
            pruned, pruned_list = prune_heads_by_kl(base_model, kl, ratio)
            pruned_per_task_ratio[(task_name, ratio)] = pruned_list
            res   = finetune_and_eval(pruned, task_name, cfg, tokenized, tokenizer)
            score = res.get(cfg["primary_key"], 0.0)
            print(f"    {cfg['primary_label']}: {score:.4f}  (baseline: {baseline_score:.4f})")
            rows.append(dict(task=task_name, metric="KL-R", level="head",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

        for ratio in PRUNE_RATIOS:
            print(f"\n  [Block] prune_ratio={ratio}")
            pruned, skipped = prune_blocks_by_kl(base_model, kl, ratio)
            print(f"    Skipped blocks: {skipped}")
            res   = finetune_and_eval(pruned, task_name, cfg, tokenized, tokenizer)
            score = res.get(cfg["primary_key"], 0.0)
            print(f"    {cfg['primary_label']}: {score:.4f}")
            rows.append(dict(task=task_name, metric="KL-R", level="block",
                             prune_ratio=ratio, primary_score=score,
                             baseline_score=baseline_score,
                             score_drop=baseline_score - score, **res))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "02_KLR_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    print("\n" + "="*70)
    print("SUMMARY TABLE — KL-R Metric")
    print("="*70)
    summary = df[["task", "level", "prune_ratio", "primary_score",
                  "baseline_score", "score_drop"]].copy()
    summary["prune_ratio"] = (summary["prune_ratio"] * 100).astype(int).astype(str) + "%"
    summary["primary_score"]  = summary["primary_score"].round(4)
    summary["baseline_score"] = summary["baseline_score"].round(4)
    summary["score_drop"]     = summary["score_drop"].round(4)
    print(summary.to_string(index=False))

    print("\nGenerating figures...")
    plot_klr_score_heatmaps(kl_per_task)
    plot_accuracy_curves(df, "head")
    plot_accuracy_curves(df, "block")
    plot_head_vs_block(df)
    plot_pruned_head_map(pruned_per_task_ratio)
    plot_score_distribution(kl_per_task)
    plot_kl_matrix_heatmap(kl_per_task)

    print("\nFigures produced:")
    for name in [
        "02_KLR_score_heatmap.png",
        "02_KLR_head_accuracy_curves.png",
        "02_KLR_block_accuracy_curves.png",
        "02_KLR_head_vs_block.png",
        "02_KLR_pruned_head_map.png",
        "02_KLR_score_distribution.png",
        "02_KLR_mean_per_layer.png",
    ]:
        print(f"  {name}")

if __name__ == "__main__":
    main()
