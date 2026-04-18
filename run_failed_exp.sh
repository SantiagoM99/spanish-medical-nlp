#   nohup bash run_failed_exp.sh > failed_exp.log 2>&1 &

set -e

WANDB_PROJECT_NER="bionlp2026-ner"
WANDB_PROJECT_ML="bionlp2026-multilabel"
export WANDB_TAGS="v7-rerun-failed"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# =============================================================================
# PART 1: RigoBERTa-2.0 (correct ID) — both tasks parallel on GPU 1 + 3
# =============================================================================
log "===== PART 1: RigoBERTa-2.0 encoders (2 runs parallel) ====="

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type encoder --model_name IIC/RigoBERTa-2.0 \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-rigoberta2" &
P1=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type encoder --model_name IIC/RigoBERTa-2.0 \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-rigoberta2-ta" &
P2=$!

wait $P1 || log "WARNING: enc-rigoberta2 failed"
wait $P2 || log "WARNING: enc-rigoberta2-ta failed"
log "===== PART 1 COMPLETE ====="

# =============================================================================
# PART 2: Gemma ML — parallel on GPU 1 + 3 (9B fits in 46GB with 4-bit)
# =============================================================================
log "===== PART 2: Gemma ML (2 runs parallel) ====="

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-gemma9b-zero-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --text_mode title_abstract --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-gemma9b-few-ta" &
P2=$!

wait $P1 || log "WARNING: dec-gemma9b-zero-ta failed"
wait $P2 || log "WARNING: dec-gemma9b-few-ta failed"
log "===== PART 2 COMPLETE ====="

# =============================================================================
# PART 3: PEFT — parallel on GPU 1 + 3, two at a time
# =============================================================================
log "===== PART 3a: PEFT NER Qwen + Llama (parallel) ====="

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-qwen7b-qlora" &
P1=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type peft --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-llama8b-qlora" &
P2=$!

wait $P1 || log "WARNING: peft-qwen7b-qlora failed"
wait $P2 || log "WARNING: peft-llama8b-qlora failed"
log "===== PART 3a COMPLETE ====="

log "===== PART 3b: PEFT ML Qwen + Llama (parallel) ====="

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-qwen7b-qlora-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type peft --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-llama8b-qlora-ta" &
P2=$!

wait $P1 || log "WARNING: peft-qwen7b-qlora-ta failed"
wait $P2 || log "WARNING: peft-llama8b-qlora-ta failed"
log "===== ALL 8 RERUNS COMPLETE ====="