# Model And Data Release Notes

This GitHub repo contains code, reports, metrics, and paper assets only.

## Dataset

The experiments use a Pix2Fact-derived clean asset package. The dataset/images are not included in this GitHub-core export.

The reproduction command template expects these files to exist locally:

```text
outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl
outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl
outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl
outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl
outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl
outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl
```

Do not commit raw images, downloaded zip packages, or private data paths to GitHub.

## Checkpoints

The main checkpoint used in the paper is:

```text
outputs/dagig_paper_main_v1/checkpoints/
  paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60
```

The LoRA adapter weight file is about 149MB, which exceeds GitHub's ordinary 100MB file limit. It is intentionally not included.

Recommended release options:

- Hugging Face model repository;
- GitHub Releases with Git LFS;
- cloud storage link with checksums.

## Main Reward Choice

The main 3B/7B comparable method uses `paper_main_v1` reward:

- format credit
- visual credit
- query credit
- evidence credit
- answer credit
- leak/path penalties

Do not use verifier-shaped `reward_v3` as the 7B mainline reward; it is a separate optional ablation.
