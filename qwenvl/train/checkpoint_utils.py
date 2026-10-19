# -*- coding: utf-8 -*-
"""Test-time model-path resolution.

checkpoint-N dirs hold only trainable weights; the latest one is rebuilt into a loadable
model (base + LoRA merged), cached, and converted by a single process under a file lock.
"""
import json
import logging
import os
import time
from pathlib import Path

import torch

WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors",
                "model.safetensors.index.json", "pytorch_model.bin.index.json")
DONE_MARKER = ".convert_done"


def is_loadable_model_dir(directory) -> bool:
    """Whether the dir has config.json and a weight file (loadable via from_pretrained)."""
    if not directory or not os.path.isdir(directory):
        return False
    if not os.path.isfile(os.path.join(directory, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(directory, f)) for f in WEIGHT_FILES)


def find_latest_checkpoint(directory):
    """Return the checkpoint-N dir with the largest step, or None."""
    if not directory or not os.path.isdir(directory):
        return None
    ckpts = []
    for p in Path(directory).glob("checkpoint-*"):
        if p.is_dir() and p.name.split("-")[-1].isdigit():
            ckpts.append((int(p.name.split("-")[-1]), p))
    return max(ckpts)[1] if ckpts else None


def convert_checkpoint_to_model(checkpoint_dir, output_dir, model_name_or_path=None, lora_ckpt=None):
    """Rebuild a checkpoint-N as in training (base + old LoRA, new LoRA, trainable weights) and save it merged."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

    ckpt_dir, out_dir = Path(checkpoint_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base model and old LoRA must match training
    meta_path = ckpt_dir.parent / "resume_meta.json"
    meta = json.load(open(meta_path)) if meta_path.is_file() else {}
    base = model_name_or_path or meta.get("model_name_or_path")
    lora = lora_ckpt or meta.get("lora_ckpt")
    if not base:
        raise ValueError(f"Cannot determine base model path: {meta_path} does not exist and model_name_or_path was not passed explicitly")
    logging.info(f"[convert] base={base} | lora_ckpt={lora}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=False)

    # Training starting point: base model + merged old LoRA
    model = video_SALMONN2_plus.from_pretrained(base, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    if lora and lora != "No":
        audio_layers = model.audio.layers
        del model.audio.layers              # keep PEFT from matching same-named projections in Whisper
        model = PeftModel.from_pretrained(model, lora)
        model.model.audio.layers = audio_layers
        model = model.merge_and_unload()

    # Architecture flags recorded in resume_meta.json (e.g. grounding_boundary_heads)
    for key, val in (meta.get("model_config") or {}).items():
        setattr(model.config, key, val)
    if meta.get("model_config"):
        logging.info(f"[convert] model_config from resume_meta: {meta['model_config']}")

    # Grow embeddings to the checkpoint vocab if tokens were added during training
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logging.info(f"[convert] resize embeddings {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)} (tokenizer has newly added tokens)")
        model.resize_token_embeddings(len(tokenizer))

    # Special token ids (all three required)
    for token, attr in (("<G>", "evi_token_id"), ("<S>", "evi_end_token_id"),
                        ("<shift>", "shift_token_id")):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"tokenizer is missing special token {token}")
        setattr(model.config, attr, tid)

    # Evidence inference config, restored from training_args.bin
    ta_path = ckpt_dir / "training_args.bin"
    train_args = torch.load(ta_path, map_location="cpu", weights_only=False) if ta_path.is_file() else None
    evidence_cfg = {
        "use_evidence": True, "use_attention_head": True, "vision_audio_layer_index": 6,
        "audio_mean_weight": 1.0, "save_gpu_trick": True, "use_flashattention": False,
    }
    for key, default in evidence_cfg.items():
        setattr(model.config, key, getattr(train_args, key, default) if train_args is not None else default)

    # Attach the training LoRA structure; readout-only checkpoints (no adapter_config.json) load into the plain model
    has_lora = (ckpt_dir / "adapter_config.json").is_file()
    if has_lora:
        acfg = json.load(open(ckpt_dir / "adapter_config.json"))
        lora_config = LoraConfig(
            r=acfg["r"], lora_alpha=acfg["lora_alpha"], target_modules=acfg["target_modules"],
            lora_dropout=acfg.get("lora_dropout", 0.05), bias=acfg.get("bias", "none"),
            task_type="CAUSAL_LM", modules_to_save=acfg.get("modules_to_save") or [],
        )
        audio_layers = model.audio.layers
        del model.audio.layers
        model = get_peft_model(model, lora_config)
        model.model.audio.layers = audio_layers
    else:
        logging.info("[convert] no adapter_config.json: readout-only checkpoint, loading head weights into the plain model")

    # DeepSpeed "latest" tag, falling back to the largest global_step* dir
    latest_file = ckpt_dir / "latest"
    if latest_file.is_file():
        tag = latest_file.read_text().strip()
    else:
        tags = sorted(
            (p for p in ckpt_dir.glob("global_step*") if p.is_dir()),
            key=lambda p: int(p.name.replace("global_step", "") or 0),
        )
        if not tags:
            raise FileNotFoundError(f"{ckpt_dir} has neither a latest tag nor any global_step* directory")
        tag = tags[-1].name
        logging.warning(f"[convert] {ckpt_dir.name} is missing the latest tag, falling back to {tag}")
    states_path = ckpt_dir / tag / "mp_rank_00_model_states.pt"
    if not states_path.is_file():
        raise FileNotFoundError(f"weight file does not exist: {states_path}")
    trainable = torch.load(states_path, map_location="cpu", weights_only=False)["module"]
    result = model.load_state_dict(trainable, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint does not match model structure: {result.unexpected_keys[:5]}")
    logging.info(f"[convert] loaded {len(trainable)} trainable tensors from {states_path.name}")

    # Merge LoRA and save in inference format
    if has_lora:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.config.save_pretrained(str(out_dir))
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        emb, head = model.get_input_embeddings(), model.get_output_embeddings()
        if emb is not None and head is not None and head.weight.data_ptr() == emb.weight.data_ptr():
            state.pop("lm_head.weight", None)   # tied: save one copy; from_pretrained re-ties it on load
    torch.save(state, out_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(str(out_dir))
    src_pre = Path(base) / "preprocessor_config.json"
    if src_pre.is_file():
        (out_dir / "preprocessor_config.json").write_bytes(src_pre.read_bytes())
    logging.info(f"[convert] saved to {out_dir}")
    return str(out_dir)


def _convert_with_lock(checkpoint_dir, converted_dir, wait_timeout=7200, poll=20):
    """Convert under a file lock so only one test process converts; the others wait and reuse."""
    converted = Path(converted_dir)
    done = converted / DONE_MARKER
    if done.is_file() and is_loadable_model_dir(str(converted)):
        logging.info(f"[convert] reusing already converted model: {converted}")
        return str(converted)

    converted.mkdir(parents=True, exist_ok=True)
    lock = converted.parent / f"{converted.name}.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        owner = True
    except FileExistsError:
        # Take over a stale lock
        owner = (time.time() - lock.stat().st_mtime > wait_timeout) if lock.is_file() else False
        if owner:
            logging.warning(f"[convert] found stale lock {lock}, taking over conversion")

    if owner:
        try:
            convert_checkpoint_to_model(checkpoint_dir, str(converted))
            done.write_text("ok")
        finally:
            lock.unlink(missing_ok=True)
        return str(converted)

    logging.info(f"[convert] another process is converting, waiting for {converted} ...")
    waited = 0
    while waited < wait_timeout:
        if done.is_file() and is_loadable_model_dir(str(converted)):
            logging.info(f"[convert] wait finished, reusing {converted}")
            return str(converted)
        time.sleep(poll)
        waited += poll
    raise TimeoutError(f"timed out waiting for checkpoint conversion ({wait_timeout}s): {converted}")


def resolve_test_model_path(model_name_or_path, output_dir, auto_convert=True):
    """Return a loadable model dir for testing.

    Tries model_name_or_path, then output_dir: a full model dir is used as is; otherwise its latest
    checkpoint-N is converted into <run_dir>/eval_models/<ckpt_name>/ (not named checkpoint-* so
    checkpoint rotation ignores it).
    """
    for candidate in (model_name_or_path, output_dir):
        if not candidate:
            continue
        if is_loadable_model_dir(candidate):
            return candidate
        ckpt = find_latest_checkpoint(candidate)
        if ckpt is not None:
            if not auto_convert:
                raise ValueError(f"{candidate} only contains checkpoints and auto conversion is disabled")
            logging.info(f"[resolve] {candidate} has no full model, using {ckpt.name} with the largest step")
            return _convert_with_lock(str(ckpt), str(ckpt.parent / "eval_models" / ckpt.name))
    raise FileNotFoundError(
        f"No loadable model found: neither model_name_or_path={model_name_or_path!r} nor output_dir={output_dir!r} "
        f"contains a full model (config.json + weights) or any checkpoint-N")
