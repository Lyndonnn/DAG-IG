# Reproduction command templates for DAG-IG paper-main runs.
#
# These commands are intentionally not marked executable by this patch.
# Running the training commands can overwrite existing output directories if
# paths are reused. Copy to a new output root when preserving current artifacts.

PROJECT_ROOT=/root/autodl-tmp/search-test-1
cd "$PROJECT_ROOT"

MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
FORMAT_ADAPTER=outputs/dagig_grpo_main/checkpoints/format_sft
TRAIN_FILE=outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl
DEV_FILE=outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl
TEST_FILE=outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl
TRAIN_CORPUS=outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl
TRAIN_CORPUS_GOLDFIXED=outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl
EVAL_CORPUS=outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl
OUT_ROOT=outputs/dagig_paper_main_v1

COMMON_TRAIN_ARGS="--model_name_or_path $MODEL --init_adapter_path $FORMAT_ADAPTER --variant paper_main_v1 --attn_impl sdpa --max_steps 60 --max_samples 320 --num_generations 4 --learning_rate 1e-6 --gradient_accumulation_steps 4 --bf16 --gradient_checkpointing --kl_coef 0.1 --max_seq_length 8192 --max_new_tokens 96 --reader_max_new_tokens 48 --temperature 0.8 --top_p 0.95 --top_k 5 --two_stage_rollout --two_stage_loss_scope stage1 --save_steps 20 --logging_steps 1"

# Main seed42 run.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/02_train_grpo.py \
  --train_file "$TRAIN_FILE" \
  --corpus_path "$TRAIN_CORPUS" \
  --output_dir outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320 \
  --seed 42 \
  $COMMON_TRAIN_ARGS

# Seed43 confirmation run.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/02_train_grpo.py \
  --train_file "$TRAIN_FILE" \
  --corpus_path "$TRAIN_CORPUS" \
  --output_dir outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43 \
  --seed 43 \
  $COMMON_TRAIN_ARGS

# Goldfixed control run.
CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/02_train_grpo.py \
  --train_file "$TRAIN_FILE" \
  --corpus_path "$TRAIN_CORPUS_GOLDFIXED" \
  --output_dir outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320 \
  --seed 42 \
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

# Seed42 main evaluation.
SEED42=outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$SEED42" \
  --model_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60 \
  --reader_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60 \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED42" \
  --reader_adapter_path "$SEED42" \
  --model_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60 \
  --reader_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60 \
  --split test \
  $COMMON_EVAL_ARGS

# Seed43 confirmation evaluation.
SEED43=outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/checkpoint-60

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$SEED43" \
  --model_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60 \
  --reader_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60 \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$SEED43" \
  --reader_adapter_path "$SEED43" \
  --model_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60 \
  --reader_tag paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60 \
  --split test \
  $COMMON_EVAL_ARGS

# Goldfixed control evaluation.
GOLDFIXED=outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$DEV_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$GOLDFIXED" \
  --reader_adapter_path "$GOLDFIXED" \
  --model_tag paper_main_v1_goldfixed_scale60_s320_ckpt60 \
  --reader_tag paper_main_v1_goldfixed_scale60_s320_ckpt60 \
  --split dev \
  $COMMON_EVAL_ARGS

CUDA_VISIBLE_DEVICES=0 python scripts/dagig_grpo/05_eval_two_stage.py \
  --eval_file "$TEST_FILE" \
  --corpus_path "$EVAL_CORPUS" \
  --adapter_path "$GOLDFIXED" \
  --reader_adapter_path "$GOLDFIXED" \
  --model_tag paper_main_v1_goldfixed_scale60_s320_ckpt60 \
  --reader_tag paper_main_v1_goldfixed_scale60_s320_ckpt60 \
  --split test \
  $COMMON_EVAL_ARGS

# Regenerate reports and paper assets after metrics/predictions exist.
python scripts/dagig_paper_main/25_consolidate_main_results.py
python scripts/dagig_paper_main/26_analyze_node_credit_components.py
python scripts/dagig_paper_main/27_build_paper_experiment_package.py
python scripts/dagig_paper_main/28_build_paper_case_studies.py
