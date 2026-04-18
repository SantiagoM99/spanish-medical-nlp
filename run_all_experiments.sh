set -e

WANDB_PROJECT_NER="bionlp2026-ner"
WANDB_PROJECT_ML="bionlp2026-multilabel"
export WANDB_TAGS="v6-best-config"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

wait_all() {
    for pid in "$@"; do
        wait $pid || log "WARNING: PID $pid failed with exit code $?"
    done
}

# =============================================================================
# PHASE 1: NER ENCODERS (7 models × 1 config = 7 runs)
# =============================================================================
log "===== PHASE 1: NER Encoders (7 runs) ====="

# Batch 1: BETO + XLM-R-base + BSC bioclinical
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type encoder --model_name dccuchile/bert-base-spanish-wwm-cased \
    --num_epochs 10 --batch_size 16 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-beto" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type encoder --model_name xlm-roberta-base \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-xlmr" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type encoder --model_name PlanTL-GOB-ES/roberta-base-biomedical-clinical-es \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-bsc-bioclinical" &
P3=$!

wait_all $P1 $P2 $P3

# Batch 2: RigoBERTa-2 + RigoBERTa-Clinical + BERTIN
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type encoder --model_name IIC/RigoBERTa-2-base \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-rigoberta2" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type encoder --model_name IIC/RigoBERTa-Clinical \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-rigoberta-clinical" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type encoder --model_name bertin-project/bertin-roberta-base-spanish \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-bertin" &
P3=$!

wait_all $P1 $P2 $P3

# Batch 3: XLM-R-large (grad accum: bs=8 × accum=2 = effective 16)
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type encoder --model_name FacebookAI/xlm-roberta-large \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "enc-xlmr-large" &
P1=$!

log "===== PHASE 1 DISPATCHED ====="

# =============================================================================
# PHASE 2: ML ENCODERS (6 models × 1 config = 6 runs)
# =============================================================================
log "===== PHASE 2: ML Encoders (6 runs) ====="

# Run ML in parallel with NER XLM-R-large (uses different GPUs)
CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type encoder --model_name dccuchile/bert-base-spanish-wwm-cased \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-beto-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type encoder --model_name PlanTL-GOB-ES/roberta-base-biomedical-es \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-bsc-biomed-ta" &
P3=$!

wait_all $P1 $P2 $P3

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type encoder --model_name IIC/RigoBERTa-2-base \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 2e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-rigoberta2-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type encoder --model_name xlm-roberta-base \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-xlmr-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type encoder --model_name bertin-project/bertin-roberta-base-spanish \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 16 --learning_rate 3e-5 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-bertin-ta" &
P3=$!

wait_all $P1 $P2 $P3

# XLM-R-large for ML (grad accum)
CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type encoder --model_name FacebookAI/xlm-roberta-large \
    --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "enc-xlmr-large-ta" &
P1=$!

log "===== PHASE 2 DISPATCHED ====="

# =============================================================================
# PHASE 3: NER DECODERS (4 models × ~2-3 configs = 10 runs)
# =============================================================================
log "===== PHASE 3: NER Decoders (10 runs) ====="

# Qwen zero + knn (parallel with ML XLM-R-large)
CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-zero" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-knn5" &
P3=$!

wait_all $P1 $P2 $P3

# Qwen knn+verify + Llama zero + knn
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 --self_verification \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-qwen7b-knn5-verify" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-llama8b-zero" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type decoder --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-llama8b-knn5" &
P3=$!

wait_all $P1 $P2 $P3

# Mistral zero + knn + Gemma zero
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-mistral7b-zero" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type decoder --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-mistral7b-knn5" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_ner_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --prompt_strategy zero_shot \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-gemma9b-zero" &
P3=$!

wait_all $P1 $P2 $P3

# Gemma knn
CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --prompt_strategy knn_few_shot --knn_k 5 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "dec-gemma9b-knn5" &
P1=$!

log "===== PHASE 3 DISPATCHED ====="

# =============================================================================
# PHASE 4: ML DECODERS (4 models × 2 configs = 8 runs)
# =============================================================================
log "===== PHASE 4: ML Decoders (8 runs) ====="

# Qwen zero + few (parallel with NER Gemma knn)
CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-qwen7b-zero-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --text_mode title_abstract --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-qwen7b-few-ta" &
P3=$!

wait_all $P1 $P2 $P3

# Llama zero + few + Mistral zero
CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type decoder --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-llama8b-zero-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type decoder --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --text_mode title_abstract --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-llama8b-few-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-mistral7b-zero-ta" &
P3=$!

wait_all $P1 $P2 $P3

# Mistral few + Gemma zero + few
CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type decoder --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --load_in_4bit --text_mode title_abstract --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-mistral7b-few-ta" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_multilabel_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --text_mode title_abstract \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-gemma9b-zero-ta" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type decoder --model_name google/gemma-2-9b-it \
    --load_in_4bit --text_mode title_abstract --few_shot \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "dec-gemma9b-few-ta" &
P3=$!

wait_all $P1 $P2 $P3
log "===== PHASE 4 COMPLETE ====="

# =============================================================================
# PHASE 5: PEFT QLoRA (4 runs)
# =============================================================================
log "===== PHASE 5: PEFT QLoRA (4 runs) ====="

CUDA_VISIBLE_DEVICES=0 python run_ner_benchmark.py \
    --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-qwen7b-qlora" &
P1=$!

CUDA_VISIBLE_DEVICES=1 python run_ner_benchmark.py \
    --model_type peft --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --peft_method qlora \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_NER --wandb_run_name "peft-llama8b-qlora" &
P2=$!

CUDA_VISIBLE_DEVICES=3 python run_multilabel_benchmark.py \
    --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-qwen7b-qlora-ta" &
P3=$!

wait_all $P1 $P2 $P3

CUDA_VISIBLE_DEVICES=0 python run_multilabel_benchmark.py \
    --model_type peft --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit --peft_method qlora --text_mode title_abstract \
    --num_epochs 10 --batch_size 8 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32 \
    --wandb_project $WANDB_PROJECT_ML --wandb_run_name "peft-llama8b-qlora-ta"

log "===== ALL 35 EXPERIMENTS COMPLETE ====="
