# Verifying The Paper-Main Result Package

This repository includes a lightweight consistency verifier for the exported paper-main metrics.

Run:

```bash
python scripts/verify_paper_main_results.py
```

Expected output:

```text
Paper-main result verification passed.
Seed42 strict gain over Format-SFT: dev +6.1, test +6.2.
Seed42 and seed43 training health checks passed.
```

This script verifies:

- the main result CSV contains the expected Format-SFT, seed42, and seed43 rows;
- the CSV matches `paper_main_v1_consolidated_results.json`;
- seed42 and seed43 both trained successfully for 60 optimizer steps / 240 micro-steps;
- both seed runs had 2 constant-reward groups.

It does **not** rerun model inference. Full inference reproduction requires the Pix2Fact-derived assets, Qwen2.5-VL-3B base model, Format-SFT adapter, and DAG-IG adapters. See `MODEL_AND_DATA.md` and `docs/REPRODUCIBILITY_APPENDIX.md`.
