# DAG-IG Pix2Fact Pilot Environment

This pilot intentionally avoids RL/GRPO and does not require `flash-attn`.

Core packages:

```text
torch
transformers
accelerate
peft
trl
qwen-vl-utils
pillow
datasets
bitsandbytes  # optional, only needed for --qlora
```

Default attention implementation is `sdpa`. If the server already has a compatible
FlashAttention install, pass `--attn_impl flash_attention_2`; otherwise keep `sdpa`.

The current local environment has Qwen2.5-VL transformer support and `qwen-vl-utils`,
but `trl` is not installed. That means SFT smoke runs are supported here, while DPO is
schema-checked and deferred to a verified TRL/LLaMA-Factory multimodal DPO setup.

For single-process smoke tests, set `CUDA_VISIBLE_DEVICES=0`. Otherwise the vanilla
Transformers `Trainer` may use PyTorch DataParallel across all visible GPUs, which is
not a safe default for Qwen-VL image batches.
