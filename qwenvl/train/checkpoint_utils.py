# -*- coding: utf-8 -*-
"""Model-path resolution and automatic checkpoint conversion for the test stage.

At the end of training we save a full model with LoRA merged (config +
pytorch_model.bin + tokenizer) that can be loaded via from_pretrained directly.
Mid-training checkpoint-N dirs, however, only contain trainable weights
(LoRA adapter + heads/embedding inside the DeepSpeed shard); the frozen backbone
must be rebuilt from the model_name_or_path + lora_ckpt recorded in
resume_meta.json, so they cannot be loaded directly.

This module lets tests point at an output dir that is still training: it picks
the checkpoint with the largest step, converts it into a loadable model, and
caches the result for reuse. A file lock ensures only one process converts
during multi-GPU testing.
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
    """Whether this is a full model dir loadable via from_pretrained (config.json + weight file)."""
    if not directory or not os.path.isdir(directory):
        return False
    if not os.path.isfile(os.path.join(directory, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(directory, f)) for f in WEIGHT_FILES)


def find_latest_checkpoint(directory):
    """Return the checkpoint-N path with the largest step in the directory, or None."""
    if not directory or not os.path.isdir(directory):
        return None
    ckpts = []
    for p in Path(directory).glob("checkpoint-*"):
        if p.is_dir() and p.name.split("-")[-1].isdigit():
            ckpts.append((int(p.name.split("-")[-1]), p))
    return max(ckpts)[1] if ckpts else None


def convert_checkpoint_to_model(checkpoint_dir, output_dir, model_name_or_path=None, lora_ckpt=None):
    """Convert a mid-training checkpoint-N into a full model dir ready for inference.

    Reproduces the training-time model construction: base + merge old LoRA ->
    attach new LoRA structure -> load trainable weights from the checkpoint ->
    merge LoRA -> save in inference format.
    """
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

    ckpt_dir, out_dir = Path(checkpoint_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # step1: resolve base model and old LoRA (must match training, otherwise the rebuilt frozen backbone is wrong)
    meta_path = ckpt_dir.parent / "resume_meta.json"
    meta = json.load(open(meta_path)) if meta_path.is_file() else {}
    base = model_name_or_path or meta.get("model_name_or_path")
    lora = lora_ckpt or meta.get("lora_ckpt")
    if not base:
        raise ValueError(f"Cannot determine base model path: {meta_path} does not exist and model_name_or_path was not passed explicitly")
    logging.info(f"[convert] base={base} | lora_ckpt={lora}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=False)

    # step2: rebuild the training starting point = base model + merged old LoRA
    model = video_SALMONN2_plus.from_pretrained(base, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    if lora and lora != "No":
        audio_layers = model.audio.layers
        del model.audio.layers              # keep PEFT from matching same-named projections in Whisper
        model = PeftModel.from_pretrained(model, lora)
        model.model.audio.layers = audio_layers
        model = model.merge_and_unload()

    # step2.5: vocab alignment -- warm-start runs may append new tokens
    # to the end of the old vocab, so the checkpoint's embedding has more rows than
    # the base model. Resize the rebuilt model to the checkpoint's vocab size first,
    # otherwise load_state_dict fails on the shape mismatch.
    # No-op for old checkpoints (vocab size == base embedding rows).
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logging.info(f"[convert] resize embeddings {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)} (tokenizer has newly added tokens)")
        model.resize_token_embeddings(len(tokenizer))

    # step3: write special token ids into config (all four protocol tokens are required)
    for token, attr in (("<G>", "evi_token_id"), ("<S>", "evi_end_token_id"),
                        ("<shift>", "shift_token_id")):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"tokenizer is missing special token {token}")
        setattr(model.config, attr, tid)

    # step4: evidence inference config (injected by the training script and saved with config; restored here from training_args.bin)
    ta_path = ckpt_dir / "training_args.bin"
    train_args = torch.load(ta_path, map_location="cpu", weights_only=False) if ta_path.is_file() else None
    evidence_cfg = {
        "use_evidence": True, "use_attention_head": True, "use_score_head": False,
        "vision_audio_layer_index": 6, "union_aggregation": False,
        "audio_mean_weight": 1.0, "save_gpu_trick": True, "use_flashattention": False,
    }
    for key, default in evidence_cfg.items():
        setattr(model.config, key, getattr(train_args, key, default) if train_args is not None else default)

    # step5: attach a new LoRA structure identical to training, then load the checkpoint's trainable weights
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

    # Normally read the "latest" tag written by DeepSpeed; it may be missing after a
    # cross-machine sync or an interrupted save. Fall back to the global_step* dir with
    # the largest step (the weights live there; the missing tag file does not affect conversion).
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

    # step6: merge LoRA and save in inference format (same as the end-of-training save logic)
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
    """File-lock guarded conversion: in multi-GPU testing, only one of N independent processes converts; the rest wait and reuse."""
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
        # Take over if the lock is stale (holder crashed without writing the done marker)
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
    """Resolve the model directory to load for testing.

    Priority:
      1. model_name_or_path is a full model dir -> use it directly;
      2. model_name_or_path contains checkpoint-N dirs -> auto-convert the one with
         the largest step (cached at <run_dir>/eval_models/<ckpt_name>/; deliberately
         not named checkpoint-* so training's checkpoint rotation / resume scans do
         not mistake it for a real checkpoint);
      3. output_dir is handled the same way as 1 and 2 (testing in place right after training).
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
