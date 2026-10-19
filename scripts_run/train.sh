# Train PadCaptioner (audio-visual dense video captioning on Qwen2.5-VL-3B).
# Usage: bash scripts_run/train.sh --dataset TRAIN.json --model MODEL_DIR --model_base MODEL_DIR \
#            --lora_ckpt LORA_DIR --output_root OUT_DIR [--gpus 0,1,2,3] [other flags below]
# Checkpoints are written to OUTPUT_ROOT/RUN_NAME.
cd "$(cd $(dirname $0); pwd)/.."

# ======================= Paths (or pass as flags, e.g. --dataset) ==================
# DATASET     training annotation json (see DATASET.md)
# MODEL       audio-augmented Qwen2.5-VL backbone; MODEL_BASE is normally the same
# LORA_CKPT   video-SALMONN-2+ LoRA checkpoint, merged into the base at startup
# OUTPUT_ROOT parent directory for run outputs
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
READOUT_ONLY=False
LORA_R=128
LORA_TARGET=qkv              # qkv, or all (q/k/v/o + MLP)
NAN_GUARD=False              # abort at the first non-finite loss/grad_norm
SAVE_TOTAL_LIMIT=5
LORA_ALPHA=256
LORA_DROPOUT=0.05
NUM_WORKER=8
PREFETCH_FACTOR=2
PERSISTENT_WORKERS=True
NO_AUDIO=False
USE_EVIDENCE=True
USE_ATTENTION_HEAD=True
LOSS_MATCH_WEIGHT=5.0
LOSS_STOP_WEIGHT=0.0        # relative weight of the planning-stop <S> (<=1 = off)
LOSS_STOP_SYMMETRIC=False   # also upweight dvc planning <G> positions
BRANCH_INTERVAL_ATTENTION=False  # dvc branches see only their own interval's media tokens
BRANCH_INTERVAL_MARGIN=1
BRANCH_INTERVAL_AUDIO_FULL=False  # confine video only; audio fully visible
LOSS_CONTRAST_WEIGHT=0.0    # timestep InfoNCE on top of the BCE match loss (0 = off)
LOSS_CONTRAST_DETACH=True   # contrast gradients reach only the evi/frm heads
GROUNDING_TARGET_DECAY=1.0   # soft grounding target 0.5^(decay*dist)
LOSS_IOU_WEIGHT=0.0          # interval soft-IoU loss weight (0 = off)
GROUNDING_COVERAGE_TRAIN=False # coverage-consistent grounding loss (dvc)
LOSS_BOUNDARY_WEIGHT=0.0       # start/end boundary-head loss (0 = off)
GROUNDING_BOUNDARY_DECAY=1.0   # boundary-peak target 0.5^(decay*|t-boundary|)
LOSS_PLATEAU_WEIGHT=0.0        # plateau-consistency hinge on the inside curve (0 = off)
GROUNDING_PLATEAU_RATIO=0.7    # match the inference grounding threshold
LOSS_BOUNDARY_RANK_WEIGHT=0.0  # internal-peak suppression on boundary heads (0 = off)
GROUNDING_BOUNDARY_RANK_MARGIN=1.0
SELF_AGGREGATION_PROB=0.0   # scheduled sampling for <G> aggregation (0 = off; not ZeRO-3)
GROUNDING_THRESHOLD=0.8      # self-aggregation readout; match evaluation
GROUNDING_MERGE_GAP=2
GROUNDING_COVERAGE_MASK=False
GROUNDING_SNAP_WINDOW=0        # >0: snap readout edges with the boundary heads
AUDIO_MEAN_WEIGHT=1
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
        --no_lora) USE_LORA=False ;;   # heads-only training, LLM frozen
        --readout_only) READOUT_ONLY=True ;;   # also freeze the token embeddings (heads only)
        --lora_r) LORA_R="$2"; shift ;;
        --lora_target) LORA_TARGET="$2"; shift ;;
        --nan_guard) NAN_GUARD="$2"; shift ;;
        --save_total_limit) SAVE_TOTAL_LIMIT="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --output_root) OUTPUT_ROOT="$2"; shift ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --prefetch_factor) PREFETCH_FACTOR="$2"; shift ;;
        --persistent_workers) PERSISTENT_WORKERS="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --use_evidence) USE_EVIDENCE=True ;;
        --no_use_evidence) USE_EVIDENCE=False ;;
        --use_attention_head) USE_ATTENTION_HEAD=True ;;
        --no_use_attention_head) USE_ATTENTION_HEAD=False ;;
        --loss_match_weight) LOSS_MATCH_WEIGHT="$2"; shift ;;
        --loss_stop_weight) LOSS_STOP_WEIGHT="$2"; shift ;;
        --loss_stop_symmetric) LOSS_STOP_SYMMETRIC="$2"; shift ;;
        --branch_interval_attention) BRANCH_INTERVAL_ATTENTION="$2"; shift ;;
        --branch_interval_margin) BRANCH_INTERVAL_MARGIN="$2"; shift ;;
        --branch_interval_audio_full) BRANCH_INTERVAL_AUDIO_FULL="$2"; shift ;;
        --loss_contrast_weight) LOSS_CONTRAST_WEIGHT="$2"; shift ;;
        --loss_contrast_detach) LOSS_CONTRAST_DETACH="$2"; shift ;;
        --grounding_target_decay) GROUNDING_TARGET_DECAY="$2"; shift ;;
        --loss_iou_weight) LOSS_IOU_WEIGHT="$2"; shift ;;
        --grounding_coverage_train) GROUNDING_COVERAGE_TRAIN="$2"; shift ;;
        --grounding_snap_window) GROUNDING_SNAP_WINDOW="$2"; shift ;;
        --loss_boundary_weight) LOSS_BOUNDARY_WEIGHT="$2"; shift ;;
        --grounding_boundary_decay) GROUNDING_BOUNDARY_DECAY="$2"; shift ;;
        --loss_plateau_weight) LOSS_PLATEAU_WEIGHT="$2"; shift ;;
        --grounding_plateau_ratio) GROUNDING_PLATEAU_RATIO="$2"; shift ;;
        --loss_boundary_rank_weight) LOSS_BOUNDARY_RANK_WEIGHT="$2"; shift ;;
        --grounding_boundary_rank_margin) GROUNDING_BOUNDARY_RANK_MARGIN="$2"; shift ;;
        --self_aggregation_prob) SELF_AGGREGATION_PROB="$2"; shift ;;
        --grounding_threshold) GROUNDING_THRESHOLD="$2"; shift ;;
        --grounding_merge_gap) GROUNDING_MERGE_GAP="$2"; shift ;;
        --grounding_coverage_mask) GROUNDING_COVERAGE_MASK="$2"; shift ;;
        --audio_mean_weight) AUDIO_MEAN_WEIGHT="$2"; shift ;;
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

# Check required paths
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

# Pick a free port
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
        --save_total_limit $SAVE_TOTAL_LIMIT \
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
        --readout_only $READOUT_ONLY \
        --lora_r $LORA_R \
        --lora_target $LORA_TARGET \
        --nan_guard $NAN_GUARD \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --tune_mm_audio $TRAIN_AUDIO \
        --tune_mm_qformer $TRAIN_QFORMER \
        --no_audio $NO_AUDIO \
        $([ "$USE_EVIDENCE" = "True" ] && echo "--use_evidence") \
        --use_attention_head $USE_ATTENTION_HEAD \
        --loss_match_weight $LOSS_MATCH_WEIGHT \
        --loss_stop_weight $LOSS_STOP_WEIGHT \
        --loss_stop_symmetric $LOSS_STOP_SYMMETRIC \
        --branch_interval_attention $BRANCH_INTERVAL_ATTENTION \
        --branch_interval_margin $BRANCH_INTERVAL_MARGIN \
        --branch_interval_audio_full $BRANCH_INTERVAL_AUDIO_FULL \
        --loss_contrast_weight $LOSS_CONTRAST_WEIGHT \
        --loss_contrast_detach $LOSS_CONTRAST_DETACH \
        --grounding_target_decay $GROUNDING_TARGET_DECAY \
        --loss_iou_weight $LOSS_IOU_WEIGHT \
        --grounding_coverage_train $GROUNDING_COVERAGE_TRAIN \
        --grounding_snap_window $GROUNDING_SNAP_WINDOW \
        --loss_boundary_weight $LOSS_BOUNDARY_WEIGHT \
        --grounding_boundary_decay $GROUNDING_BOUNDARY_DECAY \
        --loss_plateau_weight $LOSS_PLATEAU_WEIGHT \
        --grounding_plateau_ratio $GROUNDING_PLATEAU_RATIO \
        --loss_boundary_rank_weight $LOSS_BOUNDARY_RANK_WEIGHT \
        --grounding_boundary_rank_margin $GROUNDING_BOUNDARY_RANK_MARGIN \
        --self_aggregation_prob $SELF_AGGREGATION_PROB \
        --grounding_threshold $GROUNDING_THRESHOLD \
        --grounding_merge_gap $GROUNDING_MERGE_GAP \
        --grounding_coverage_mask $GROUNDING_COVERAGE_MASK \
        --audio_mean_weight $AUDIO_MEAN_WEIGHT \
        --aggregation_jitter_ratio $AGGREGATION_JITTER \
        --use_flashattention $USE_FLASHATTENTION \
        --vision_audio_layer_index $VISION_AUDIO_LAYER_INDEX \
        --print_trainable_params False
