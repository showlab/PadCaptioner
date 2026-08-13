cd "$(cd $(dirname $0); pwd)/.."


# ======================= Paths: fill these in before running =======================
# DATASET     evaluation annotation json (same schema as training; see README.md)
# MODEL       merged model directory (pytorch_model.bin + config.json + tokenizer).
#             With USE_EVIDENCE=True this must NOT be a checkpoint-N subdirectory —
#             checkpoints only hold the LoRA adapter and DeepSpeed shards. Convert one
#             first with my_tools/convert_checkpoint_to_model.py.
# MODEL_BASE  audio-augmented Qwen2.5-VL backbone directory
# OUTPUT_ROOT parent directory for results (a per-run subdirectory is created below)
# Each can also be supplied on the command line, e.g. --model /path/to/model
DATASET=""
MODEL=""
MODEL_BASE=""
OUTPUT_ROOT=""
# ===================================================================================

RUN_NAME="padcaptioner_test"

BS=1
MAX_PIXELS=61250
MIN_PIXELS=784
MIN_FRAMES=64
MAX_FRAMES=256
INTERVAL=0.5

USE_LORA=False
LORA_CKPT=No
NUM_SAMPLE=1
DO_SAMPLE=False
NO_AUDIO=False
GROUNDING_MERGE_GAP=2
GROUNDING_SELECT=mass
USE_EVIDENCE=True
USE_FLASHATTENTION=False
GROUNDING_THRESHOLD=0.7
DEINTERLEAVE=True

export CUDA_VISIBLE_DEVICES=0
export ARNOLD_ID=0


export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_$(id -u)}
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TORCH_CPP_LOG_LEVEL=ERROR


PORT_BASE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --do_sample) DO_SAMPLE=True ;;
        --num_sample) NUM_SAMPLE="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --use_evidence) USE_EVIDENCE=True ;;
        --no_use_evidence) USE_EVIDENCE=False ;;
        --grounding_threshold) GROUNDING_THRESHOLD="$2"; shift ;;
        --grounding_merge_gap) GROUNDING_MERGE_GAP="$2"; shift ;;
        --grounding_select) GROUNDING_SELECT="$2"; shift ;;
        --port_base) PORT_BASE="$2"; shift ;;
        --no_deinterleave) DEINTERLEAVE=False ;;
        --deinterleave) DEINTERLEAVE=True ;;
        --output_root) OUTPUT_ROOT="$2"; shift ;;
        --gpus) export CUDA_VISIBLE_DEVICES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"

# Fail early with a precise message instead of letting an empty path reach torchrun
MISSING=""
[ -z "$DATASET" ]     && MISSING="${MISSING} DATASET(--dataset)"
[ -z "$MODEL" ]       && MISSING="${MISSING} MODEL(--model)"
[ -z "$MODEL_BASE" ]  && MISSING="${MISSING} MODEL_BASE(--model_base)"
[ -z "$OUTPUT_ROOT" ] && MISSING="${MISSING} OUTPUT_ROOT(--output_root)"
if [ -n "$MISSING" ]; then
    echo "[test] FAILED: unset path(s):${MISSING}" >&2
    echo "[test] Set them at the top of this script or pass them as flags." >&2
    exit 1
fi
if [ ! -f "$DATASET" ]; then
    echo "[test] FAILED: dataset not found: $DATASET" >&2
    exit 1
fi

# Per-run working directory: input shards + logs + result files all live here
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
SPLIT_DIR=${RUN_DIR}/splits
mkdir -p "$SPLIT_DIR"

# If unset, auto-pick NGPUS consecutive free ports to avoid clashes between concurrent tests
if [ -z "$PORT_BASE" ]; then
    PORT_BASE=$(python3 - "$NGPUS" <<'PYEOF'
import random, socket, sys
n = int(sys.argv[1])
def free(p):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p)); return True
        except OSError:
            return False
for _ in range(200):
    b = random.randint(20000, 60000)
    if all(free(b + i) for i in range(n)):
        print(b); break
else:
    print(12800)
PYEOF
)
fi
echo "[test] RUN_NAME=${RUN_NAME} NGPUS=${NGPUS} GPUS=${CUDA_VISIBLE_DEVICES} PORT_BASE=${PORT_BASE}"


rm -f ${OUTPUT_ROOT}/${RUN_NAME}/test_results_rank*.json


if ! python3 scripts/split_data.py "$DATASET" $NGPUS "${SPLIT_DIR}/"; then
    echo "[test] FAILED: split_data.py failed, aborting" >&2
    exit 1
fi

LOG_DIR=${OUTPUT_ROOT}/${RUN_NAME}/logs
mkdir -p ${LOG_DIR}

PIDS=()
for i in $(seq 0 $((${NGPUS}-1))); do
    TARGET_GPU=${GPU_LIST[$i]}
    if [ "$i" -eq 0 ]; then SHARD_LOG=/dev/stdout; else SHARD_LOG=${LOG_DIR}/rank${i}.log; fi
    CUDA_VISIBLE_DEVICES=$TARGET_GPU torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$((PORT_BASE + i + 8 * ARNOLD_ID)) \
        qwenvl/train/train_qwen_modified.py \
            --model_base $MODEL_BASE \
            --run_test True \
            --pred_rank $((NGPUS * ARNOLD_ID + i)) \
            --deepspeed scripts/zero2.json \
            --model_name_or_path "$MODEL" \
            --dataset_use ${SPLIT_DIR}/$((NGPUS * ARNOLD_ID + i)).json \
            --bf16 \
            --output_dir ${OUTPUT_ROOT} \
            --per_device_train_batch_size $BS \
            --max_pixels $MAX_PIXELS \
            --min_pixels $MIN_PIXELS \
            --video_max_frame_pixels $MAX_PIXELS \
            --video_min_frame_pixels $MIN_PIXELS \
            --eval_strategy "no" \
            --save_strategy "no" \
            --logging_steps 1 \
            --model_max_length 131072 \
            --dataloader_num_workers 8 \
            --run_name $RUN_NAME \
            --report_to none \
            --video_min_frames $MIN_FRAMES \
            --video_max_frames $MAX_FRAMES \
            --base_interval $INTERVAL \
            --use_lora $USE_LORA \
            --lora_ckpt $LORA_CKPT \
            --num_sample $NUM_SAMPLE \
            --do_sample $DO_SAMPLE \
            --no_audio $NO_AUDIO \
            --grounding_threshold $GROUNDING_THRESHOLD \
            --grounding_merge_gap $GROUNDING_MERGE_GAP \
            --grounding_select $GROUNDING_SELECT \
            --use_flashattention $USE_FLASHATTENTION \
            --deinterleave_parallel_output $DEINTERLEAVE \
            $([ "$USE_EVIDENCE" = "True" ] && echo "--use_evidence" || echo "") > "$SHARD_LOG" 2>&1 &
    PIDS+=($!)
done
[ "$NGPUS" -gt 1 ] && echo "[test] shard 1-$((NGPUS-1)) logs: ${LOG_DIR}/rank*.log"

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        if [ "$i" -eq 0 ]; then
            echo "[test] FAILED: rank $i (GPU ${GPU_LIST[$i]}) exited non-zero" >&2
        else
            echo "[test] FAILED: rank $i (GPU ${GPU_LIST[$i]}) exited non-zero, see ${LOG_DIR}/rank${i}.log" >&2
        fi
        FAILED=$((FAILED + 1))
    fi
done


NRESULT=$(ls ${OUTPUT_ROOT}/${RUN_NAME}/test_results_rank*.json 2>/dev/null | wc -l)
if [ "$NRESULT" -ne "$NGPUS" ]; then
    echo "[test] FAILED: expected ${NGPUS} result shards, got ${NRESULT}" >&2
    FAILED=$((FAILED + 1))
fi

if [ "$FAILED" -ne 0 ]; then
    echo "[test] ${FAILED} failure(s); results are incomplete, do not use for analysis" >&2
    exit 1
fi
echo "All done. Results: ${OUTPUT_ROOT}/${RUN_NAME}/test_results_rank*.json"
