"""
00_finetune_base.py

Step 1 of the experiment pipeline.
Fine-tunes BERT-base on SST-2, CoLA, and MRPC, saves checkpoints,
and produces visualizations for the final report:

  figures/00_training_curves.png           — loss + metric vs epoch, per task
  figures/00_base_metrics_summary.png      — bar chart of final eval metrics
  figures/00_baseline_entropy_heatmap.png  — per-head attention entropy heatmap
  figures/00_entropy_per_layer.png         — mean entropy per layer, all tasks
  results/00_base_results.csv             — raw numbers

Run once before any metric script:
    python 00_finetune_base.py

Resume after an interrupted run (latest checkpoint per task under checkpoints/bert-<task>/ next to this script):
    python 00_finetune_base.py --resume

Skip tasks that already finished (best model saved under checkpoints/bert-<task>/best/):
    python 00_finetune_base.py --skip-if-best
"""

import argparse
import os
import random
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import evaluate

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(_SCRIPT_DIR, "figures")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
for d in [CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

TASK_CONFIG = {
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric_name": "glue/sst2",
        "primary_key": "eval_accuracy",
        "primary_label": "Accuracy",
        "epochs": 3,
        "color": "#2563EB",
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric_name": "glue/cola",
        "primary_key": "eval_matthews_correlation",
        "primary_label": "Matthews Corr.",
        "epochs": 3,
        "color": "#16A34A",
    },
    "mrpc": {
        "dataset": ("glue", "mrpc"),
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric_name": "glue/mrpc",
        "primary_key": "eval_f1",
        "primary_label": "F1",
        "epochs": 3,
        "color": "#DC2626",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAYERS = 12
N_HEADS = 12
CALIB_SIZE = 256

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_tokenize_fn(tokenizer, text_keys):
    col1, col2 = text_keys

    def tokenize(batch):
        if col2 is None:
            return tokenizer(batch[col1], truncation=True)
        return tokenizer(batch[col1], batch[col2], truncation=True)

    return tokenize


def compute_metrics_fn(metric_name):
    metric = evaluate.load(*metric_name.split("/"))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    return compute_metrics


# ---------------------------------------------------------------------------
# Baseline attention entropy (for heatmap figures)
# ---------------------------------------------------------------------------
def compute_baseline_entropy(model, dataloader):
    model.eval().to(DEVICE)
    entropy_sum = np.zeros((N_LAYERS, N_HEADS))
    count = 0
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
# Figure 1 — Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(all_log_histories):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "BERT-base Fine-tuning Curves on GLUE Tasks", fontsize=14, fontweight="bold"
    )

    for ax, (task_name, cfg) in zip(axes, TASK_CONFIG.items()):
        df = pd.DataFrame(all_log_histories[task_name])
        color = cfg["color"]

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

        ax2 = ax.twinx()

        if not train_df.empty and "epoch" in train_df.columns:
            ax.plot(
                train_df["epoch"],
                train_df["loss"],
                color=color,
                lw=1.4,
                alpha=0.45,
                label="Train loss",
            )
        if not eval_df.empty and "epoch" in eval_df.columns:
            ax.plot(
                eval_df["epoch"],
                eval_df["eval_loss"],
                color=color,
                lw=2.2,
                ls="--",
                label="Val loss",
            )
            pk = cfg["primary_key"]
            if pk in eval_df.columns:
                ax2.plot(
                    eval_df["epoch"],
                    eval_df[pk],
                    color="#F59E0B",
                    lw=2,
                    marker="o",
                    ms=6,
                    label=cfg["primary_label"],
                )

        ax.set_title(task_name.upper(), fontweight="bold", color=color, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=color)
        ax2.set_ylabel(cfg["primary_label"], color="#F59E0B")
        ax2.spines["right"].set_visible(True)
        lines1, l1 = ax.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "00_training_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Base model performance bar chart
# ---------------------------------------------------------------------------
def plot_base_metrics(final_results):
    tasks = list(final_results.keys())
    values = [final_results[t]["primary_value"] for t in tasks]
    labels = [TASK_CONFIG[t]["primary_label"] for t in tasks]
    colors = [TASK_CONFIG[t]["color"] for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        tasks, values, color=colors, width=0.45, edgecolor="white", linewidth=1.5
    )
    for bar, val, lbl in zip(bars, values, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.4f}\n({lbl})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(
        "Baseline BERT-base Performance — GLUE Tasks", fontweight="bold", fontsize=13
    )
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=12)

    path = os.path.join(FIGURES_DIR, "00_base_metrics_summary.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Baseline attention entropy heatmaps
# ---------------------------------------------------------------------------
def plot_entropy_heatmaps(entropy_per_task):
    n = len(entropy_per_task)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    vmin = min(e.min() for e in entropy_per_task.values())
    vmax = max(e.max() for e in entropy_per_task.values())

    for ax, (task_name, entropy) in zip(axes, entropy_per_task.items()):
        im = sns.heatmap(
            entropy,
            ax=ax,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            xticklabels=[f"H{h}" for h in range(N_HEADS)],
            yticklabels=[f"L{l}" for l in range(N_LAYERS)],
            cbar=(ax is axes[-1]),
            linewidths=0.25,
            linecolor="white",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
        )
        ax.set_title(
            f"{task_name.upper()} — Mean Attention Entropy",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=7, rotation=0)

    fig.suptitle(
        "Baseline Per-Head Attention Entropy  (higher = more diffuse)",
        fontsize=12,
        y=1.01,
    )
    path = os.path.join(FIGURES_DIR, "00_baseline_entropy_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Mean entropy per layer (all tasks)
# ---------------------------------------------------------------------------
def plot_entropy_per_layer(entropy_per_task):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for task_name, entropy in entropy_per_task.items():
        ax.plot(
            range(N_LAYERS),
            entropy.mean(axis=1),
            marker="o",
            lw=2,
            ms=6,
            color=TASK_CONFIG[task_name]["color"],
            label=task_name.upper(),
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean attention entropy")
    ax.set_title(
        "Mean Attention Entropy per Layer — Baseline BERT-base",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xticks(range(N_LAYERS))
    ax.legend()
    path = os.path.join(FIGURES_DIR, "00_entropy_per_layer.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-base on SST-2, CoLA, and MRPC (GLUE)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume each task from the latest checkpoint in its output_dir "
            "when present; otherwise train from pretrained weights."
        ),
    )
    parser.add_argument(
        "--skip-if-best",
        action="store_true",
        dest="skip_if_best",
        help=(
            "If checkpoints/bert-<task>/best exists, skip training for that task "
            "and load it for eval metrics and entropy figures only."
        ),
    )
    cli = parser.parse_args()

    all_log_histories = {}
    final_results = {}
    entropy_per_task = {}

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"  Fine-tuning BERT on {task_name.upper()}")
        print(f"{'=' * 60}\n")
        set_seed(SEED)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        raw = load_dataset(*cfg["dataset"])
        tokenized = raw.map(make_tokenize_fn(tokenizer, cfg["text_keys"]), batched=True)
        collator = DataCollatorWithPadding(tokenizer)

        out_dir = os.path.join(CHECKPOINT_DIR, f"bert-{task_name}")
        best_dir = os.path.join(out_dir, "best")
        skip = cli.skip_if_best and os.path.isdir(best_dir)

        resume_from = None
        if not skip and cli.resume:
            resume_from = (
                get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
            )
            if resume_from:
                print(f"  Resuming from: {resume_from}\n")
            else:
                print(
                    f"  --resume: no checkpoint in {out_dir}; training from scratch.\n"
                )

        if skip:
            print(f"  --skip-if-best: using existing {best_dir} (no training).\n")
            model = AutoModelForSequenceClassification.from_pretrained(best_dir)
            all_log_histories[task_name] = []

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=SEED,
            data_seed=SEED,
            report_to="none",
            fp16=torch.cuda.is_available(),
            logging_steps=20,
        )

        if not skip:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=cfg["num_labels"]
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics_fn(cfg["metric_name"]),
        )

        if not skip:
            trainer.train(resume_from_checkpoint=resume_from)
            all_log_histories[task_name] = trainer.state.log_history
            trainer.save_model(best_dir)
            tokenizer.save_pretrained(best_dir)

        eval_out = trainer.evaluate()
        pk = cfg["primary_key"]
        final_results[task_name] = {
            "primary_value": eval_out.get(pk, 0.0),
            **eval_out,
        }
        print(f"\n  Final eval: {eval_out}")

        # Baseline entropy
        drop_cols = [
            c
            for c in tokenized["train"].column_names
            if c not in ("input_ids", "attention_mask", "token_type_ids", "label")
        ]
        calib = tokenized["train"].select(range(CALIB_SIZE)).remove_columns(drop_cols)
        loader = torch.utils.data.DataLoader(calib, batch_size=32, collate_fn=collator)
        entropy_per_task[task_name] = compute_baseline_entropy(trainer.model, loader)

    # ---- Figures ----
    print("\nGenerating figures...")
    plot_training_curves(all_log_histories)
    plot_base_metrics(final_results)
    plot_entropy_heatmaps(entropy_per_task)
    plot_entropy_per_layer(entropy_per_task)

    # ---- CSV ----
    rows = [{"task": t, **v} for t, v in final_results.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "00_base_results.csv"), index=False
    )

    print(f"\nAll outputs in {FIGURES_DIR}/ and {RESULTS_DIR}/")
    print(
        "Figures:  00_training_curves.png | 00_base_metrics_summary.png |",
        "00_baseline_entropy_heatmap.png | 00_entropy_per_layer.png",
    )


if __name__ == "__main__":
    main()
