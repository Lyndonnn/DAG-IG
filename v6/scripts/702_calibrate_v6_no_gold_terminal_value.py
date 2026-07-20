#!/usr/bin/env python3
"""Cross-fit and audit a deployable no-gold terminal success value."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


FEATURE_NAMES = (
    "support_logit",
    "reader_candidate_mean_logprob",
    "minimum_support_reader",
    "support_reader_interaction",
    "answer_token_length",
    "is_unknown",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_no_gold_match", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source, target = normalized(text).split(), normalized(phrase).split()
    return bool(target and any(source[index : index + len(target)] == target for index in range(len(source) - len(target) + 1)))


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def feature(row: dict[str, Any]) -> list[float]:
    support = max(-20.0, min(20.0, float(row["support_logit"])))
    reader = max(-20.0, min(0.0, float(row["reader_candidate_mean_logprob"])))
    return [
        support,
        reader,
        min(support, reader),
        support * reader / 20.0,
        min(40.0, max(0.0, float(row["answer_token_length"]))),
        float(bool(row["is_unknown"])),
    ]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int) -> dict[str, np.ndarray]:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - center) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    weights[0] = math.log(prevalence / (1.0 - prevalence))
    for _ in range(steps):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(y)
        hessian[1:, 1:] += np.eye(design.shape[1] - 1) * l2
        hessian += np.eye(design.shape[1]) * 1e-7
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1e-7:
            break
    return {"center": center, "scale": scale, "weights": weights}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [np.ones((len(x), 1)), (x - model["center"]) / model["scale"]], axis=1
    )
    return sigmoid(design @ model["weights"])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5
    return float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores < high if index + 1 < bins else scores <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return result


def metric(scores: np.ndarray, labels: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    brier = float(np.mean((scores - labels) ** 2))
    base_brier = float(np.mean((baseline - labels) ** 2))
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "base_rate": float(labels.mean()),
        "auc": auc(scores, labels),
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_improvement": base_brier - brier,
        "ece_10bin": ece(scores, labels),
        "probability_mean": float(scores.mean()),
        "probability_min": float(scores.min()),
        "probability_max": float(scores.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN":
        raise ValueError("no-gold terminal protocol is not frozen")
    if freeze["code_hashes"]["calibrator"] != sha256(Path(__file__).resolve()):
        raise ValueError("no-gold terminal calibrator changed after freeze")
    for key in ("source_freeze", "answer_actions", "evidence_actions", "private_labels", "corpus", "eval_utils", "source_scoring_freeze"):
        if sha256(Path(freeze["input_paths"][key])) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen no-gold terminal input changed: {key}")
    manifests = [read_json(Path(path)) for path in freeze["input_paths"]["score_manifests"]]
    for path, expected in zip(freeze["input_paths"]["score_manifests"], freeze["input_hashes"]["score_manifests"]):
        if sha256(Path(path)) != expected:
            raise ValueError("terminal score manifest changed")
    scores: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        score_path = Path(manifest["output_paths"]["scores"])
        if sha256(score_path) != manifest["output_hashes"]["scores"]:
            raise ValueError("terminal score shard changed")
        scores.update({str(row["answer_action_id"]): row for row in read_jsonl(score_path)})

    answers = read_jsonl(Path(freeze["input_paths"]["answer_actions"]))
    evidence = {str(row["evidence_action_id"]): row for row in read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))}
    labels = {str(row["sample_id"]): row for row in read_jsonl(Path(freeze["input_paths"]["private_labels"]))}
    corpus = {str(row["doc_id"]): row for row in read_jsonl(Path(freeze["input_paths"]["corpus"]))}
    helper = load_module(Path(freeze["input_paths"]["eval_utils"]))
    if len(answers) != int(freeze["answer_actions"]) or len(scores) != len(answers):
        raise ValueError("expanded answer/score universe differs")
    train_ids = set(freeze["policy_train_sample_ids"])
    development_ids = set(freeze["development_sample_ids"])
    records = []
    for action in answers:
        answer_id = str(action["answer_action_id"])
        score = scores[answer_id]
        parent = evidence[str(action["evidence_action_id"])]
        label = labels[str(action["sample_id"])]
        partition = "policy_train" if action["sample_id"] in train_ids else "internal_holdout" if action["sample_id"] in development_ids else "invalid"
        if partition == "invalid":
            raise ValueError("answer action outside frozen partition")
        match = helper.answer_match_details(action["candidate_answer"], label["gold_answer"], label.get("aliases") or [])
        accepted = [label["gold_answer"], *(label.get("aliases") or [])]
        text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in parent["selected_docs"])
        selected_urls = {canonical_url(doc.get("url")) for doc in parent["selected_docs"]}
        positive_urls = {
            canonical_url((corpus.get(str(doc_id)) or {}).get("final_url") or (corpus.get(str(doc_id)) or {}).get("url"))
            for doc_id in label.get("positive_doc_ids") or []
        }
        support = bool(positive_urls & selected_urls) or any(phrase_contains(text, value) for value in accepted)
        records.append(
            {
                "answer_action_id": answer_id,
                "evidence_action_id": str(action["evidence_action_id"]),
                "query_id": str(action["query_id"]),
                "sample_id": str(action["sample_id"]),
                "partition": partition,
                "feature": feature(score),
                "strict_proxy": int(bool(match["answer_correct"] and support)),
                "answer_correct_proxy": bool(match["answer_correct"]),
                "evidence_support_proxy": bool(support),
            }
        )

    x = np.asarray([row["feature"] for row in records], dtype=np.float64)
    y = np.asarray([row["strict_proxy"] for row in records], dtype=np.float64)
    train_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    development_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    config = freeze["calibration"]
    fold_by_sample = {
        sample_id: int(hashlib.sha256(f"{config['fold_seed_text']}:{sample_id}".encode()).hexdigest(), 16) % int(config["folds"])
        for sample_id in train_ids
    }
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    baselines = np.full(len(records), np.nan, dtype=np.float64)
    folds = []
    for fold in range(int(config["folds"])):
        fit_index = np.asarray([index for index in train_index if fold_by_sample[records[index]["sample_id"]] != fold])
        validation_index = np.asarray([index for index in train_index if fold_by_sample[records[index]["sample_id"]] == fold])
        model = fit_logistic(x[fit_index], y[fit_index], float(config["l2"]), int(config["max_newton_steps"]))
        probabilities[validation_index] = predict(model, x[validation_index])
        baselines[validation_index] = float(y[fit_index].mean())
        folds.append(
            {
                "fold": fold,
                "fit_samples": len({records[index]["sample_id"] for index in fit_index}),
                "validation_samples": len({records[index]["sample_id"] for index in validation_index}),
                "fit_actions": len(fit_index),
                "validation_actions": len(validation_index),
                "fit_positives": int(y[fit_index].sum()),
            }
        )
    final_model = fit_logistic(x[train_index], y[train_index], float(config["l2"]), int(config["max_newton_steps"]))
    probabilities[development_index] = predict(final_model, x[development_index])
    baselines[development_index] = float(y[train_index].mean())
    low, high = config["probability_clip"]
    probabilities = np.clip(probabilities, float(low), float(high))
    if not np.isfinite(probabilities).all() or not np.isfinite(baselines).all():
        raise ValueError("incomplete no-gold terminal probabilities")

    oof_metrics = metric(probabilities[train_index], y[train_index], baselines[train_index])
    development_metrics = metric(probabilities[development_index], y[development_index], baselines[development_index])
    by_evidence: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(records, probabilities):
        by_evidence[row["evidence_action_id"]].append(float(probability))
    nonconstant = float(np.mean([max(values) - min(values) > 1e-5 for values in by_evidence.values()]))
    specs = freeze["gates_spec"]
    gates = {
        "complete_expanded_actions": len(records) == int(freeze["answer_actions"]),
        "train_sample_grouped_oof_complete": np.isfinite(probabilities[train_index]).all(),
        "oof_auc": oof_metrics["auc"] >= float(specs["oof_auc_min"]),
        "oof_brier_improvement": oof_metrics["brier_improvement"] >= float(specs["oof_brier_improvement_vs_fold_constant_min"]),
        "development_auc": development_metrics["auc"] >= float(specs["development_auc_min"]),
        "development_brier_improvement": development_metrics["brier_improvement"] >= float(specs["development_brier_improvement_vs_train_base_rate_min"]),
        "development_ece": development_metrics["ece_10bin"] <= float(specs["development_ece_max"]),
        "nonconstant_answer_groups": nonconstant >= float(specs["nonconstant_group_rate_min"]),
        "equivalence_logit_not_used": True,
        "runtime_features_contain_no_gold_or_qrels": True,
        "development_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    decision = "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value_path = output / "v6_no_gold_terminal_success_values.jsonl"
    private_path = output / "v6_no_gold_terminal_private_audit.jsonl"
    model_path = output / "v6_no_gold_terminal_calibrator.json"
    with value_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(
                json.dumps(
                    {
                        "answer_action_id": row["answer_action_id"],
                        "evidence_action_id": row["evidence_action_id"],
                        "query_id": row["query_id"],
                        "sample_id": row["sample_id"],
                        "partition": row["partition"],
                        "terminal_success_probability": float(probability),
                        "terminal_log_value": math.log(float(probability)),
                        "calibration_source": "sample_grouped_crossfit" if row["partition"] == "policy_train" else "policy_train_fit_development_score",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    with private_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(json.dumps({**row, "terminal_success_probability": float(probability)}, sort_keys=True) + "\n")
    serial_model = {
        "feature_names": list(FEATURE_NAMES),
        "center": final_model["center"].tolist(),
        "scale": final_model["scale"].tolist(),
        "weights": final_model["weights"].tolist(),
        "fit_partition": "policy_train_only",
        "folds": folds,
        "equivalence_logit_used": False,
    }
    model_path.write_text(json.dumps(serial_model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": {
            "oof": oof_metrics,
            "development": development_metrics,
            "nonconstant_answer_group_rate": nonconstant,
            "answer_actions": len(records),
            "evidence_groups": len(by_evidence),
        },
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"terminal_values": str(value_path), "private_audit": str(private_path), "calibrator": str(model_path)},
        "output_hashes": {"terminal_values": sha256(value_path), "private_audit": sha256(private_path), "calibrator": sha256(model_path)},
        "gold_or_qrels_in_runtime_features": False,
        "equivalence_logit_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
