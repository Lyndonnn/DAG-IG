# Reproduction command templates for corrected DAG-IG paper-main runs.
#
# These commands are intentionally not marked executable by this patch.
# Running the training commands can overwrite existing output directories if
# paths are reused. Copy to a new output root when preserving current artifacts.

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
cd "$PROJECT_ROOT"

MODEL=${DAGIG_LOCAL_3B_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
FORMAT_ADAPTER=outputs/dagig_grpo_main/checkpoints/format_sft
TRAIN_FILE=outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl
DEV_FILE=outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl
TEST_FILE=outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl
TRAIN_CORPUS=outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl
EVAL_CORPUS=outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl
OUT_ROOT=outputs/dagig_paper_main_v1

COMMON_TRAIN_ARGS="--model_name_or_path $MODEL --init_adapter_path $FORMAT_ADAPTER --variant paper_main_v1 --attn_impl sdpa --max_steps 60 --max_samples 320 --num_generations 4 --learning_rate 1e-6 --gradient_accumulation_steps 4 --bf16 --gradient_checkpointing --kl_coef 0.1 --max_seq_length 8192 --max_new_tokens 96 --reader_max_new_tokens 48 --temperature 0.8 --top_p 0.95 --top_k 5 --two_stage_rollout --two_stage_loss_scope stage1 --save_steps 20 --logging_steps 1"

# KL-fixed seed42 run.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/02_train_grpo.py \
  --train_file "$TRAIN_FILE" \
  --corpus_path "$TRAIN_CORPUS" \
  --output_dir outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42 \
  --seed 42 \
  $COMMON_TRAIN_ARGS

# KL-fixed seed43 run.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/02_train_grpo.py \
  --train_file "$TRAIN_FILE" \
  --corpus_path "$TRAIN_CORPUS" \
  --output_dir outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43 \
  --seed 43 \
  $COMMON_TRAIN_ARGS

COMMON_EVAL_ARGS="--model_name_or_path $MODEL --output_root $OUT_ROOT --attn_impl sdpa --stage1_max_new_tokens 96 --reader_max_new_tokens 48 --top_k 5 --bf16"

# Format-SFT baseline, own reader.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$FORMAT_ADAPTER" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag format_sft_two_stage_own_full \
  --reader_tag format_sft_two_stage_own_full \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$FORMAT_ADAPTER" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag format_sft_two_stage_own_full \
  --reader_tag format_sft_two_stage_own_full \
  --split test \
  $COMMON_EVAL_ARGS

# KL-fixed seed42 own-reader evaluation.
SEED42=outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/checkpoint-60

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$SEED42" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60 \
  --reader_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60 \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$SEED42" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60 \
  --reader_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60 \
  --split test \
  $COMMON_EVAL_ARGS

# KL-fixed seed43 own-reader evaluation.
SEED43=outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/checkpoint-60

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$SEED43" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60 \
  --reader_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60 \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$SEED43" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60 \
  --reader_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60 \
  --split test \
  $COMMON_EVAL_ARGS

# Fixed-reader control: KL-fixed queries with the same Format-SFT reader.

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader \
  --reader_tag reader_format_sft \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader \
  --reader_tag reader_format_sft \
  --split test \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader \
  --reader_tag reader_format_sft \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$FORMAT_ADAPTER" \
  --model_tag paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader \
  --reader_tag reader_format_sft \
  --split test \
  $COMMON_EVAL_ARGS

# Regenerate reports and paper assets after metrics/predictions exist.
python scripts/dagig_paper_main/55_rescore_checker_v4.py
python scripts/dagig_paper_main/57_consolidate_klfixed_results.py
python scripts/dagig_paper_main/58_audit_corpus_reality.py
python scripts/verify_paper_main_results.py
