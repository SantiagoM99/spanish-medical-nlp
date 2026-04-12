# Usage:
#   nohup bash run_all_experiments.sh > experiments.log 2>&1 &

set -e

WANDB_PROJECT_NER="bionlp2026-ner"
WANDB_PROJECT_ML="bionlp2026-multilabel"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

wait_all() {
    for pid in "$@"; do
        wait $pid || log "WARNING: PID $pid failed with exit code $?"
    done
}

# =============================================================================
# BATCH 1: NER Encoders
# =============================================================================
log "===== BATCH 1/8: NER Encoders (BETO, XLM-R, BSC bioclinical) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type encoder \
    --model_name dccuchile/bert-base-spanish-wwm-cased \
    --num_epochs 5 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-beto" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type encoder \
    --model_name xlm-roberta-base \
    --num_epochs 5 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-xlmr" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type encoder \
    --model_name PlanTL-GOB-ES/roberta-base-biomedical-clinical-es \
    --num_epochs 5 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-bsc-bioclinical" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 1/8 COMPLETE ====="

# =============================================================================
# BATCH 2: NER SOTA Encoders + ML Encoder start
# =============================================================================
log "===== BATCH 2/8: NER RigoBERTa2 + RigoClinical + ML BETO ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type encoder \
    --model_name IIC/RigoBERTa-2.0 \
    --num_epochs 5 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-rigoberta2" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type encoder \
    --model_name IIC/RigoBERTa-Clinical \
    --num_epochs 5 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-rigoberta-clinical" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type encoder \
    --model_name dccuchile/bert-base-spanish-wwm-cased \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-beto-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 2/8 COMPLETE ====="

# =============================================================================
# BATCH 3: ML Encoders (3 in parallel)
# =============================================================================
log "===== BATCH 3/8: ML Encoders (BSC, RigoBERTa2, XLM-R) ====="

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type encoder \
    --model_name PlanTL-GOB-ES/roberta-base-biomedical-es \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-bsc-biomed-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type encoder \
    --model_name IIC/RigoBERTa-2.0 \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-rigoberta2-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type encoder \
    --model_name xlm-roberta-base \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-xlmr-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 3/8 COMPLETE ====="

# =============================================================================
# BATCH 4: NER Decoders (3 in parallel)
# =============================================================================
log "===== BATCH 4/8: NER Decoders (Qwen zero, Qwen knn, Llama zero) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-zero" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-knn5" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type decoder \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-llama8b-zero" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 4/8 COMPLETE ====="

# =============================================================================
# BATCH 5: NER Decoder ablation + ML Decoders
# =============================================================================
log "===== BATCH 5/8: NER Qwen knn+verify + Llama knn + ML Qwen zero ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --prompt_strategy knn_few_shot --knn_k 5 \
    --self_verification \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-knn5-verify" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-llama8b-knn5" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-qwen7b-zero-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 5/8 COMPLETE ====="

# =============================================================================
# BATCH 6: ML Decoders (3 in parallel)
# =============================================================================
log "===== BATCH 6/8: ML Decoders (Qwen few, Llama zero, Llama few) ====="

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --text_mode title_abstract \
    --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-qwen7b-few-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-llama8b-zero-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --text_mode title_abstract \
    --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-llama8b-few-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 6/8 COMPLETE ====="

# =============================================================================
# BATCH 7: PEFT (3 in parallel)
# =============================================================================
log "===== BATCH 7/8: PEFT (NER Qwen, NER Llama, ML Qwen) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type peft \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --peft_method qlora \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-qwen7b-qlora" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type peft \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --peft_method qlora \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-llama8b-qlora" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type peft \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --peft_method qlora \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-qwen7b-qlora-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== BATCH 7/8 COMPLETE ====="

# =============================================================================
# BATCH 8: Last PEFT
# =============================================================================
log "===== BATCH 8/8: PEFT ML Llama ====="

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type peft \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --peft_method qlora \
    --text_mode title_abstract \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-llama8b-qlora-ta"

log "===== BATCH 8/8 COMPLETE ====="
log "ALL 22 EXPERIMENTS COMPLETE"
