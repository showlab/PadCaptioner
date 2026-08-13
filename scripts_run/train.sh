cd "$(cd $(dirname $0); pwd)/.."

# ======================= Paths: fill these in before running =======================
# DATASET     training annotation json (schema documented in DATASET.md). A handful of
#             very long dvc samples dominate peak memory; if they cause OOM, either drop
#             them from the annotation file or lower BS and raise ACCUM_STEPS.
# MODEL       audio-augmented Qwen2.5-VL backbone directory
# MODEL_BASE  base model used for the LoRA merge (normally identical to MODEL)
# LORA_CKPT   video-SALMONN-2+ LoRA checkpoint, merged into the base at startup
# OUTPUT_ROOT parent directory for run outputs (checkpoints and the merged model)
# Each can also be supplied on the command line, e.g. --dataset /path/to/train.json
DATASET=""
MODEL=""
MODEL_BASE=""
LORA_CKPT=""
OUTPUT_ROOT=""
# ===================================================================================

LR=2e-5
BS=6
ACCUM_STEPS=4
RUN_NAME="padcaptioner_train"
DEEPSPEED=2
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
TRAIN_AUDIO=False
TRAIN_QFORMER=False
EPOCH=1
MAX_PIXELS=61250
MIN_PIXELS=784
SAVE_STEPS=30
MIN_FRAMES=64
MAX_FRAMES=256
INTERVAL=0.5
USE_LORA=True
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
TRAIN_TYPE=sft
NUM_WORKER=8
PREFETCH_FACTOR=2
PERSISTENT_WORKERS=True
NO_AUDIO=False
USE_EVIDENCE=True
USE_SCORE_HEAD=False
USE_ATTENTION_HEAD=True
LOSS_MATCH_WEIGHT=5.0
AUDIO_MEAN_WEIGHT=1
UNION_AGGREGATION=False
AGGREGATION_JITTER=0.1
USE_FLASHATTENTION=False
VISION_AUDIO_LAYER_INDEX=6


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,4,6,7}


export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_$(id -u)}
export ARNOLD_ID=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --train_audio) TRAIN_AUDIO=True ;;
        --train_qformer) TRAIN_QFORMER=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --output_root) OUTPUT_ROOT="$2"; shift ;;
        --train_type) TRAIN_TYPE="$2"; shift ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --prefetch_factor) PREFETCH_FACTOR="$2"; shift ;;
        --persistent_workers) PERSISTENT_WORKERS="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --use_evidence) USE_EVIDENCE=True ;;
        --no_use_evidence) USE_EVIDENCE=False ;;
        --use_score_head) USE_SCORE_HEAD=True ;;
        --no_use_score_head) USE_SCORE_HEAD=False ;;
        --use_attention_head) USE_ATTENTION_HEAD=True ;;
        --no_use_attention_head) USE_ATTENTION_HEAD=False ;;
        --loss_match_weight) LOSS_MATCH_WEIGHT="$2"; shift ;;
        --audio_mean_weight) AUDIO_MEAN_WEIGHT="$2"; shift ;;
        --union_aggregation) UNION_AGGREGATION="$2"; shift ;;
        --aggregation_jitter) AGGREGATION_JITTER="$2"; shift ;;
        --use_flashattention) USE_FLASHATTENTION=True ;;
        --no_use_flashattention) USE_FLASHATTENTION=False ;;
        --vision_audio_layer_index) VISION_AUDIO_LAYER_INDEX="$2"; shift ;;
        --gpus) export CUDA_VISIBLE_DEVICES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Derived after parsing so that --gpus takes effect
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# Fail early with a precise message instead of letting an empty path reach torchrun
MISSING=""
[ -z "$DATASET" ]     && MISSING="${MISSING} DATASET(--dataset)"
[ -z "$MODEL" ]       && MISSING="${MISSING} MODEL(--model)"
[ -z "$MODEL_BASE" ]  && MISSING="${MISSING} MODEL_BASE(--model_base)"
[ -z "$LORA_CKPT" ]   && MISSING="${MISSING} LORA_CKPT(--lora_ckpt)"
[ -z "$OUTPUT_ROOT" ] && MISSING="${MISSING} OUTPUT_ROOT(--output_root)"
if [ -n "$MISSING" ]; then
    echo "[train] FAILED: unset path(s):${MISSING}" >&2
    echo "[train] Set them at the top of this script or pass them as flags." >&2
    exit 1
fi
if [ ! -f "$DATASET" ]; then
    echo "[train] FAILED: dataset not found: $DATASET" >&2
    exit 1
fi

OUTPUT_DIR=${OUTPUT_ROOT}/${RUN_NAME}
mkdir -p "$OUTPUT_DIR"

# Pick a free port automatically to avoid clashing with other training/testing jobs
MASTER_PORT=$(python3 - <<'PYEOF'
import random, socket
def free(p):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p)); return True
        except OSError:
            return False
for _ in range(200):
    b = random.randint(20000, 60000)
    if free(b):
        print(b); break
else:
    print(29507)
PYEOF
)
echo "[train] RUN_NAME=${RUN_NAME} GPUS=${CUDA_VISIBLE_DEVICES} BS=${BS}x${ACCUM_STEPS} FRAMES=${MAX_FRAMES} WORKERS=${NUM_WORKER} PORT=${MASTER_PORT}"

torchrun --nproc_per_node=${NGPUS} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${MASTER_PORT} \
    qwenvl/train/train_qwen_modified.py \
        --deepspeed scripts/zero${DEEPSPEED}.json \
        --model_name_or_path "$MODEL" \
        --dataset_use $DATASET \
        --tune_mm_vision $TRAIN_ENC \
        --tune_mm_mlp $TRAIN_PROJ \
        --tune_mm_llm $TRAIN_LLM \
        --bf16 \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --max_pixels $MAX_PIXELS \
        --min_pixels $MIN_PIXELS \
        --video_max_frame_pixels $MAX_PIXELS \
        --video_min_frame_pixels $MIN_PIXELS \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --save_total_limit 5 \
        --learning_rate $LR \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 131072 \
        --gradient_checkpointing True \
        --dataloader_num_workers $NUM_WORKER \
        $([ "$NUM_WORKER" -gt 0 ] && echo "--dataloader_prefetch_factor $PREFETCH_FACTOR --dataloader_persistent_workers $PERSISTENT_WORKERS") \
        --dataloader_drop_last True \
        --run_name $RUN_NAME \
        --report_to wandb \
        --video_min_frames $MIN_FRAMES \
        --video_max_frames $MAX_FRAMES \
        --base_interval $INTERVAL \
        --model_base $MODEL_BASE \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --train_type $TRAIN_TYPE \
        --tune_mm_audio $TRAIN_AUDIO \
        --tune_mm_qformer $TRAIN_QFORMER \
        --no_audio $NO_AUDIO \
        $([ "$USE_EVIDENCE" = "True" ] && echo "--use_evidence") \
        --use_score_head $USE_SCORE_HEAD \
        --use_attention_head $USE_ATTENTION_HEAD \
        --loss_match_weight $LOSS_MATCH_WEIGHT \
        --audio_mean_weight $AUDIO_MEAN_WEIGHT \
        --union_aggregation $UNION_AGGREGATION \
        --aggregation_jitter_ratio $AGGREGATION_JITTER \
        --use_flashattention $USE_FLASHATTENTION \
        --vision_audio_layer_index $VISION_AUDIO_LAYER_INDEX \
        --print_trainable_params False
