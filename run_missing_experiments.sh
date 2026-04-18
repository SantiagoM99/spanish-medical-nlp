#!/usr/bin/env bash
# Runs the 8 decoder experiments missing from run_all_experiments.sh, so the
# decoder rows in paper Tables 4 and 6 become fully populated.
# Assumes the same Python env / GPU layout as run_all_experiments.sh.

set -e

WANDB_PROJECT_NER="bionlp2026-ner"
WANDB_PROJECT_ML="bionlp2026-multilabel"
export WANDB_TAGS="v6-best-config,missing-runs"

log()       { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
wait_all()  { for pid in "$@"; do wait $pid || log "WARNING: PID $pid failed with exit code $?"; done; }

# --- pre-flight: warn if any target GPU is tight on memory ------------------
log "GPU free memory (MiB):"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
for gpu in 0 1 3; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
    if [ "$free" -lt 22000 ]; then
        log "WARNING: GPU $gpu has only ${free}MiB free (<22GB). Consider deferring."
    fi
done

# =============================================================================
# BATCH A: Classification QLoRA (3 runs) — GPUs 0, 1, 3
# =============================================================================
log "===== BATCH A: Classification QLoRA (3 runs) ====="

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type peft --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-llama8b-qlora-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type peft --model_name google/gemma-2-9b-it \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-gemma9b-qlora-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type peft --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-mistral7b-qlora-ta" &
P3=$!

wait_all $P1 $P2 $P3

# =============================================================================
# BATCH B: NER k-NN + self-verification (3 runs) — GPUs 0, 1, 3
# =============================================================================
log "===== BATCH B: NER k-NN+verify (3 runs) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 --self_verification \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-gemma9b-knn5-verify" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 --self_verification \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-llama8b-knn5-verify" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type decoder --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 --self_verification \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-mistral7b-knn5-verify" &
P3=$!

wait_all $P1 $P2 $P3

# =============================================================================
# BATCH C: NER QLoRA (2 runs) — GPUs 0, 1
# =============================================================================
log "===== BATCH C: NER QLoRA (2 runs) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type peft --model_name google/gemma-2-9b-it \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-gemma9b-qlora" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type peft --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-mistral7b-qlora" &
P2=$!

wait_all $P1 $P2

log "===== ALL 8 MISSING EXPERIMENTS COMPLETE ====="
