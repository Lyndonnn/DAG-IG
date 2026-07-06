# Verifying The Corrected Paper-Main Result Package

This repository includes a lightweight consistency verifier for the corrected KL-fixed paper-main metrics.

Run:

```bash
python scripts/verify_paper_main_results.py
```

Expected output:

```text
Corrected KL-fixed paper-main verification passed.
Two-seed KL-fixed strict gain over Format-SFT: dev +5.1, test +4.7.
Core fixes passed: k3 KL, checker v4, and training-health checks.
```

This script verifies:

- the main result CSV contains the expected Format-SFT v4, KL-fixed seed42, KL-fixed seed43, and two-seed mean rows;
- the CSV matches `klfixed_grpo_60_summary.json`;
- seed42 and seed43 both trained successfully for 60 optimizer steps / 240 micro-steps;
- KL-fixed seed42 had `3/240` constant-reward groups and seed43 had `1/240`;
- `core_fix_validation.json` confirms k3 KL and checker-v4 behavior.

It does **not** rerun model inference. Full inference reproduction requires the Pix2Fact-derived assets, Qwen2.5-VL-3B base model, Format-SFT adapter, and DAG-IG adapters. See `MODEL_AND_DATA.md` and `docs/REPRODUCIBILITY_APPENDIX.md`.
