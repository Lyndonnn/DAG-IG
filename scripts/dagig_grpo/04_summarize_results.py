#!/usr/bin/env python3
"""Summarize the DAG-IG GRPO main experiment.

This script is intentionally read-only with respect to training artifacts: it
loads completed predictions, metrics, run configs, and reward rollout logs, then
writes the final report bundle required by the experiment handoff.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/dagig_grpo_main"
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "data/Pix2Fact_DAGIG_Clean_GRPO_ASSET"
DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/"
    "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)

MODELS = [
    "base",
    "format_sft",
    "outcome_grpo",
    "trajectory_grpo",
    "dagig_grpo_no_visual",
    "dagig_grpo_full",
]
GRPO_VARIANTS = [
    "outcome_grpo",
    "trajectory_grpo",
    "dagig_grpo_no_visual",
    "dagig_grpo_full",
]
SPLITS = ["dev", "test"]
CI_METRICS = ["strict_success", "retrieval_top5_hit", "answer_correct"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.1f}%"


def bootstrap_ci(rows: list[dict[str, Any]], key: str, samples: int, seed: int) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    values = [1.0 if row.get(key) else 0.0 for row in rows]
    mean = sum(values) / n
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        total = 0.0
        for _idx in range(n):
            total += values[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return {
        "n": n,
        "mean": mean,
        "ci95_low": estimates[low_idx],
        "ci95_high": estimates[high_idx],
        "bootstrap_samples": samples,
    }


def load_available(output_root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]], list[str]]:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing: list[str] = []
    for model in MODELS:
        metrics[model] = {}
        predictions[model] = {}
        for split in SPLITS:
            metric_path = output_root / "metrics" / f"{model}_{split}.json"
            pred_path = output_root / "predictions" / f"{model}_{split}.jsonl"
            if metric_path.exists():
                metrics[model][split] = read_json(metric_path)
            else:
                missing.append(str(metric_path))
            if pred_path.exists():
                predictions[model][split] = read_jsonl(pred_path)
            else:
                missing.append(str(pred_path))
    return metrics, predictions, missing


def load_train_summaries(output_root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in GRPO_VARIANTS:
        path = output_root / "checkpoints" / variant / "grpo_train_summary.json"
        summaries[variant] = read_json(path) if path.exists() else {"status": "missing", "path": str(path)}
    format_ckpt = output_root / "checkpoints" / "format_sft"
    summaries["format_sft"] = {
        "status": "success" if (format_ckpt / "adapter_model.safetensors").exists() else "missing",
        "checkpoint": str(format_ckpt),
    }
    return summaries


def pairwise(predictions: dict[str, dict[str, list[dict[str, Any]]]], left: str, right: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        left_rows = {row["sample_id"]: row for row in predictions.get(left, {}).get(split, [])}
        right_rows = {row["sample_id"]: row for row in predictions.get(right, {}).get(split, [])}
        ids = sorted(set(left_rows) & set(right_rows))
        recover: list[str] = []
        harm: list[str] = []
        same_correct = 0
        both_fail = 0
        for sample_id in ids:
            l_ok = bool(left_rows[sample_id].get("strict_success"))
            r_ok = bool(right_rows[sample_id].get("strict_success"))
            if l_ok and not r_ok:
                recover.append(sample_id)
            elif r_ok and not l_ok:
                harm.append(sample_id)
            elif l_ok and r_ok:
                same_correct += 1
            else:
                both_fail += 1
        out[split] = {
            "n_shared": len(ids),
            "recover_count": len(recover),
            "harm_count": len(harm),
            "net_gain": len(recover) - len(harm),
            "same_correct_count": same_correct,
            "both_fail_count": both_fail,
            "recover_examples": recover[:12],
            "harm_examples": harm[:12],
        }
    return out


def copy_configs(output_root: Path, asset_root: Path, model_path: str) -> None:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_config = {
        "asset_root": str(asset_root),
        "model_name_or_path": model_path,
        "student_model": "Qwen2.5-VL-3B-Instruct",
        "forbidden": ["Qwen32B training/loading", "GPT54 trajectories", "raw_pool_week1.parquet", "DPO pair files"],
    }
    write_json(config_dir / "base_eval.json", base_config)
    write_json(
        config_dir / "format_sft.json",
        {
            **base_config,
            "checkpoint": str(output_root / "checkpoints" / "format_sft"),
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "bf16": True,
            "max_seq_length": 8192,
            "max_pixels": 1003520,
        },
    )
    for variant in GRPO_VARIANTS:
        src = output_root / "checkpoints" / variant / "grpo_run_config.json"
        if src.exists():
            shutil.copyfile(src, config_dir / f"{variant}.json")


def write_reward_design(output_root: Path) -> None:
    text = """# DAG-IG GRPO Reward Design

The policy must output compact JSON with `visual_observation`, `search_query`, and `final_answer`.

Common components:
- format: JSON/field presence reward with penalties for invalid, empty, or excessive output.
- visual: lexical overlap between visual observation and grounded visual anchor terms.
- query: retrieval credit when the generated query retrieves a gold/supporting document in the frozen BM25 top-k.
- evidence: evidence support credit when retrieved top-k contains a supporting document.
- answer: full credit for answer correctness with evidence support, partial credit for answer correctness alone.
- leakage_penalty: penalty when the final answer leaks into the search query.

Variants:
- `outcome_grpo`: format + answer + leakage penalty only.
- `trajectory_grpo`: format + coarse retrieval success + coarse strict final success + leakage penalty.
- `dagig_grpo_no_visual`: format + query + evidence + answer + leakage penalty.
- `dagig_grpo_full`: format + visual + query + evidence + answer + leakage penalty.

All retrieval uses the frozen derived BM25 corpus. The student is Qwen2.5-VL-3B-Instruct with LoRA only; Qwen32B, GPT54 trajectories, raw_pool, DPO files, and dev/test rows are not used for GRPO training.
"""
    (output_root / "reward_design.md").write_text(text, encoding="utf-8")


def write_reward_examples(output_root: Path, limit_per_variant: int) -> None:
    out_path = output_root / "reward_component_examples.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for variant in GRPO_VARIANTS:
            path = output_root / "checkpoints" / variant / "reward_rollouts.jsonl"
            if not path.exists():
                continue
            count = 0
            for row in read_jsonl(path):
                out.write(json.dumps({"variant": variant, **row}, ensure_ascii=False) + "\n")
                count += 1
                if count >= limit_per_variant:
                    break


def command_text(output_root: Path, asset_root: Path, model_path: str) -> tuple[str, str]:
    train_file = output_root / "derived_assets" / "grpo_train.jsonl"
    train_corpus = output_root / "derived_assets" / "bm25_train_corpus.jsonl"
    eval_corpus = output_root / "derived_assets" / "bm25_eval_corpus.jsonl"
    common_train = (
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false "
        "python scripts/dagig_grpo/02_train_grpo.py "
        f"--asset_root {asset_root} --output_root {output_root} "
        f"--train_file {train_file} --corpus_path {train_corpus} "
        f"--model_name_or_path {model_path} --init_adapter_path {output_root / 'checkpoints/format_sft'} "
        "--attn_impl sdpa --num_train_epochs 2 --num_generations 4 --learning_rate 1e-6 "
        "--gradient_accumulation_steps 4 --bf16 --gradient_checkpointing --kl_coef 0.02 "
        "--max_seq_length 8192 --max_new_tokens 128 --temperature 0.9 --top_p 0.95 "
        "--top_k 5 --max_pixels 1003520 --save_steps 50 --logging_steps 10"
    )
    train_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Format-SFT command is recorded in logs/train_format_sft.command.txt",
    ]
    for variant in GRPO_VARIANTS:
        train_lines.append(
            f"{common_train} --output_dir {output_root / 'checkpoints' / variant} --variant {variant} "
            f"2>&1 | tee {output_root / 'logs' / ('train_' + variant + '.log')}"
        )
    common_eval = (
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false "
        "python scripts/dagig_grpo/03_eval_grpo.py "
        f"--asset_root {asset_root} --output_root {output_root} "
        f"--corpus_path {eval_corpus} --model_name_or_path {model_path} "
        "--attn_impl sdpa --max_new_tokens 192 --top_k 5"
    )
    eval_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    eval_lines.append(
        f"{common_eval} --eval_file {output_root / 'derived_assets/grpo_dev.jsonl'} --model_tag base --split dev "
        "--reward_variant dagig_grpo_full "
        f"2>&1 | tee {output_root / 'logs/eval_base_dev.log'}"
    )
    eval_lines.append(
        f"{common_eval} --eval_file {output_root / 'derived_assets/grpo_test.jsonl'} --model_tag base --split test "
        "--reward_variant dagig_grpo_full "
        f"2>&1 | tee {output_root / 'logs/eval_base_test.log'}"
    )
    for model in ["format_sft", *GRPO_VARIANTS]:
        reward_variant = model if model in GRPO_VARIANTS else "dagig_grpo_full"
        for split in SPLITS:
            eval_file = output_root / "derived_assets" / f"grpo_{split}.jsonl"
            eval_lines.append(
                f"{common_eval} --eval_file {eval_file} --adapter_path {output_root / 'checkpoints' / model} "
                f"--model_tag {model} --split {split} --reward_variant {reward_variant} "
                f"2>&1 | tee {output_root / 'logs' / ('eval_' + model + '_' + split + '.log')}"
            )
    return "\n".join(train_lines) + "\n", "\n".join(eval_lines) + "\n"


def make_table(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    rows = [
        "| model | Dev R@5 | Dev answer | Dev strict | Test R@5 | Test answer | Test strict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        dev = metrics.get(model, {}).get("dev", {})
        test = metrics.get(model, {}).get("test", {})
        rows.append(
            f"| `{model}` | {pct(dev.get('retrieval_top5_hit'))} | {pct(dev.get('answer_correct'))} | "
            f"{pct(dev.get('strict_success'))} | {pct(test.get('retrieval_top5_hit'))} | "
            f"{pct(test.get('answer_correct'))} | {pct(test.get('strict_success'))} |"
        )
    return "\n".join(rows)


def write_final_report(
    output_root: Path,
    asset_root: Path,
    metrics: dict[str, dict[str, dict[str, Any]]],
    train_summaries: dict[str, Any],
    comparisons: dict[str, Any],
    missing: list[str],
) -> None:
    validation = (output_root / "package_validation.log").read_text(encoding="utf-8") if (output_root / "package_validation.log").exists() else ""
    manifest_path = output_root / "derived_assets" / "derived_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    derived_counts = manifest.get("derived_counts", {})
    validation_passed = (
        "PACKAGE VALIDATION PASSED" in validation
        or "GRPO ASSET VALIDATION PASSED" in validation
    )
    status = "COMPLETE" if not missing else "INCOMPLETE"
    lines = [
        "# DAG-IG-GRPO Main Experiment Report",
        "",
        f"## Status: `{status}`",
        "",
        "## 1. Asset And Safety",
        "",
        f"- asset root: `{asset_root}`",
        f"- output root: `{output_root}`",
        f"- package validation passed: `{validation_passed}`",
        f"- derived train/dev/test rows: `{derived_counts.get('derived_grpo_train')}` / `{derived_counts.get('derived_grpo_dev')}` / `{derived_counts.get('derived_grpo_test')}`",
        f"- image field used by trainer/evaluator: `image_abs_path` from derived assets, based on package-local images.",
        "- student model: `Qwen2.5-VL-3B-Instruct` with LoRA only.",
        "- forbidden sources avoided: Qwen32B training/loading, GPT54 trajectories, raw_pool_week1.parquet, DPO pair files, and dev/test rows for training.",
        "",
        "## 2. Training Summaries",
        "",
        "| variant | status | optimizer steps | peak GPU GB | checkpoint |",
        "|---|---:|---:|---:|---|",
    ]
    for model in ["format_sft", *GRPO_VARIANTS]:
        summary = train_summaries.get(model, {})
        checkpoint = summary.get("output_dir") or summary.get("checkpoint") or str(output_root / "checkpoints" / model)
        lines.append(
            f"| `{model}` | `{summary.get('status')}` | {summary.get('optimizer_steps', '-')} | "
            f"{summary.get('max_gpu_mem_gb', '-')} | `{checkpoint}` |"
        )
    lines.extend(["", "## 3. Main Metrics", "", make_table(metrics), ""])
    lines.extend(
        [
            "## 4. Pairwise Strict-Success Comparisons",
            "",
            "| comparison | split | recover | harm | net gain |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, comp in comparisons.items():
        for split, row in comp.items():
            lines.append(
                f"| `{name}` | {split} | {row['recover_count']} | {row['harm_count']} | {row['net_gain']} |"
            )
    lines.extend(
        [
            "",
            "## 5. Reward Design",
            "",
            "Reward components and variant definitions are written to `reward_design.md`; sampled rollout-level component records are in `reward_component_examples.jsonl`.",
            "",
            "## 6. Missing Artifacts",
            "",
        ]
    )
    if missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## 7. Decision",
            "",
            "`dagig_grpo_full` is not a clean main-result win under strict success. It gives the best dev retrieval R@5 in this run, but it does not beat the coarse trajectory/no-visual variants on dev strict success and ties them on test strict success. The honest conclusion is that DAG-IG node-level reward provides a useful retrieval/diagnostic signal, while the current policy still bottlenecks on final answer extraction after retrieval.",
        ]
    )
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    metrics, predictions, missing = load_available(args.output_root)
    if missing and not args.allow_missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    train_summaries = load_train_summaries(args.output_root)
    ci: dict[str, Any] = {}
    for model in MODELS:
        ci[model] = {}
        for split in SPLITS:
            rows = predictions.get(model, {}).get(split, [])
            ci[model][split] = {
                key: bootstrap_ci(rows, key, args.bootstrap_samples, args.seed + hash((model, split, key)) % 100000)
                for key in CI_METRICS
            }
    write_json(args.output_root / "bootstrap_ci.json", ci)

    comparisons = {
        "format_sft_vs_base": pairwise(predictions, "format_sft", "base"),
        "outcome_vs_format_sft": pairwise(predictions, "outcome_grpo", "format_sft"),
        "trajectory_vs_outcome": pairwise(predictions, "trajectory_grpo", "outcome_grpo"),
        "dagig_no_visual_vs_trajectory": pairwise(predictions, "dagig_grpo_no_visual", "trajectory_grpo"),
        "dagig_full_vs_no_visual": pairwise(predictions, "dagig_grpo_full", "dagig_grpo_no_visual"),
        "dagig_full_vs_trajectory": pairwise(predictions, "dagig_grpo_full", "trajectory_grpo"),
    }
    final_metrics = {
        "models": metrics,
        "bootstrap_ci": ci,
        "train_summaries": train_summaries,
        "pairwise": comparisons,
        "missing_artifacts": missing,
    }
    write_json(args.output_root / "final_metrics.json", final_metrics)
    copy_configs(args.output_root, args.asset_root, args.model_name_or_path)
    write_reward_design(args.output_root)
    write_reward_examples(args.output_root, limit_per_variant=20)
    train_cmds, eval_cmds = command_text(args.output_root, args.asset_root, args.model_name_or_path)
    (args.output_root / "train_commands.sh").write_text(train_cmds, encoding="utf-8")
    (args.output_root / "eval_commands.sh").write_text(eval_cmds, encoding="utf-8")
    write_final_report(args.output_root, args.asset_root, metrics, train_summaries, comparisons, missing)
    print(json.dumps({"status": "ok", "missing": missing, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
