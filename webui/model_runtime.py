import io
import os
import re
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

import torch
from dotenv import set_key

from airllm import AutoModel

from .config import (
    DEVICE_DEFAULT,
    ENV_FILE,
    HF_TOKEN,
    MODEL_BASE_DIR_DEFAULT,
    MODEL_ID_DEFAULT,
    MODEL_PATH_DEFAULT,
)
from .model_catalog import validate_local_model_source

_model = None
_model_lock = threading.Lock()
_runtime = {
    "model_id": MODEL_ID_DEFAULT,
    "model_path": MODEL_PATH_DEFAULT,
    "model_base_dir": MODEL_BASE_DIR_DEFAULT,
    "device": DEVICE_DEFAULT,
    "model_source": None,
}

_load_jobs = {}
_load_jobs_lock = threading.Lock()

_SHARD_RE = re.compile(r"loading shard\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LAYERS_RE = re.compile(r"running layers[^\r\n]*?(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_COUNT_RE = re.compile(r"(?<![\d:])(\d{1,5})\s*/\s*(\d{1,5})(?![\d:])")
_PERCENT_RE = re.compile(r"(\d{1,3})%")


def normalize_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value if value else None


def normalize_device(value: Optional[str]) -> Optional[str]:
    value = normalize_optional(value)
    if not value:
        return None
    value = value.split()[0]
    if value == "cuda":
        return "cuda:0"
    return value


def resolve_model_source(
    model_id: str,
    model_path: Optional[str],
    model_base_dir: Optional[str],
) -> str:
    model_path = normalize_optional(model_path)
    model_base_dir = normalize_optional(model_base_dir)
    if model_path:
        return model_path
    if model_base_dir:
        return os.path.join(model_base_dir, *model_id.split("/"))
    return model_id


def get_runtime_field(key: str):
    with _model_lock:
        return _runtime.get(key)


def resolve_requested_source(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_base_dir: Optional[str] = None,
):
    with _model_lock:
        target_model_id = normalize_optional(model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
        target_model_path = normalize_optional(model_path)
        if target_model_path is None:
            target_model_path = _runtime["model_path"]
        target_model_base_dir = normalize_optional(model_base_dir)
        if target_model_base_dir is None:
            target_model_base_dir = _runtime["model_base_dir"]
        target_model_source = resolve_model_source(
            target_model_id,
            target_model_path,
            target_model_base_dir,
        )

    return {
        "model_id": target_model_id,
        "model_path": target_model_path,
        "model_base_dir": target_model_base_dir,
        "model_source": target_model_source,
    }


def runtime_state():
    with _model_lock:
        return {
            "model_loaded": _model is not None,
            "model_id": _runtime["model_id"],
            "model_source": _runtime["model_source"]
            or resolve_model_source(
                _runtime["model_id"],
                _runtime["model_path"],
                _runtime["model_base_dir"],
            ),
            "model_path": _runtime["model_path"],
            "model_base_dir": _runtime["model_base_dir"],
            "device": _runtime["device"],
        }


def apply_settings(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_base_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    with _model_lock:
        target_model_id = normalize_optional(model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
        target_model_path = normalize_optional(model_path)
        if target_model_path is None:
            target_model_path = _runtime["model_path"]
        target_model_base_dir = normalize_optional(model_base_dir)
        if target_model_base_dir is None:
            target_model_base_dir = _runtime["model_base_dir"]
        target_device = normalize_device(device) or _runtime["device"] or DEVICE_DEFAULT
        target_model_source = resolve_model_source(
            target_model_id,
            target_model_path,
            target_model_base_dir,
        )
        _runtime.update(
            {
                "model_id": target_model_id,
                "model_path": target_model_path,
                "model_base_dir": target_model_base_dir,
                "device": target_device,
                "model_source": target_model_source,
            }
        )


def persist_runtime_to_env():
    with _model_lock:
        snapshot = {
            "model_id": _runtime["model_id"] or "",
            "model_path": _runtime["model_path"] or "",
            "model_base_dir": _runtime["model_base_dir"] or "",
            "device": _runtime["device"] or "",
        }

    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), "AIRLLM_MODEL_ID", snapshot["model_id"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_PATH", snapshot["model_path"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_BASE_DIR", snapshot["model_base_dir"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_DEVICE", snapshot["device"], quote_mode="never")


def _cleanup_model():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_base_dir: Optional[str] = None,
    device: Optional[str] = None,
    force_reload: bool = False,
):
    global _model

    with _model_lock:
        target_model_id = normalize_optional(model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
        target_model_path = normalize_optional(model_path)
        if target_model_path is None:
            target_model_path = _runtime["model_path"]
        target_model_base_dir = normalize_optional(model_base_dir)
        if target_model_base_dir is None:
            target_model_base_dir = _runtime["model_base_dir"]
        target_device = normalize_device(device) or _runtime["device"] or DEVICE_DEFAULT
        target_model_source = resolve_model_source(
            target_model_id,
            target_model_path,
            target_model_base_dir,
        )

        same_config = (
            _model is not None
            and _runtime["model_id"] == target_model_id
            and _runtime["model_path"] == target_model_path
            and _runtime["model_base_dir"] == target_model_base_dir
            and _runtime["device"] == target_device
            and _runtime["model_source"] == target_model_source
        )
        if same_config and not force_reload:
            return _model

        if _model is not None:
            _model = None
            _cleanup_model()

        kwargs = {"device": target_device}
        if HF_TOKEN:
            kwargs["hf_token"] = HF_TOKEN
        if os.path.exists(target_model_source):
            validate_local_model_source(target_model_source)

        _model = AutoModel.from_pretrained(target_model_source, **kwargs)
        _runtime.update(
            {
                "model_id": target_model_id,
                "model_path": target_model_path,
                "model_base_dir": target_model_base_dir,
                "device": target_device,
                "model_source": target_model_source,
            }
        )

    return _model


def apply_download_result(model_id: str, base_dir: str, target_dir: str, set_as_active_model: bool = True):
    global _model

    with _model_lock:
        _runtime["model_base_dir"] = base_dir
        if set_as_active_model:
            if _model is not None:
                _model = None
                _cleanup_model()
            _runtime.update(
                {
                    "model_id": model_id,
                    "model_path": target_dir,
                    "model_source": target_dir,
                }
            )


def _new_load_job(payload: dict):
    job_id = str(uuid.uuid4())
    with _load_jobs_lock:
        _load_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Queued",
            "current": None,
            "total": None,
            "logs": ["Load job queued."],
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
            "model_id": payload.get("model_id"),
            "model_path": payload.get("model_path"),
            "model_base_dir": payload.get("model_base_dir"),
            "device": payload.get("device"),
            "persist_env": bool(payload.get("persist_env", False)),
            "result": None,
        }
    return job_id


def _load_job_update(job_id: str, **fields):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def _load_job_log(job_id: str, message: str):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return
        logs = job.get("logs", [])
        logs.append(message)
        if len(logs) > 300:
            logs = logs[-300:]
        job["logs"] = logs


def load_job_snapshot(job_id: str):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return None
        return dict(job)


def active_load_job():
    with _load_jobs_lock:
        active = [job for job in _load_jobs.values() if job.get("status") in ("queued", "loading")]
        if not active:
            return {"job_id": None, "status": "idle"}
        active.sort(key=lambda item: item.get("started_at", 0), reverse=True)
        job = active[0]
        return {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "model_id": job.get("model_id"),
            "started_at": job.get("started_at"),
            "stage": job.get("stage"),
            "progress": job.get("progress"),
            "message": job.get("message"),
        }


def _update_load_progress(job_id: str, *, stage: Optional[str] = None, current: Optional[int] = None,
                          total: Optional[int] = None, progress: Optional[int] = None, message: Optional[str] = None):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return

        old_stage = job.get("stage")
        old_progress = int(job.get("progress") or 0)

        if stage is not None:
            job["stage"] = stage
        if current is not None:
            job["current"] = int(current)
        if total is not None:
            job["total"] = int(total)

        if progress is not None:
            next_progress = max(0, min(100, int(progress)))
            if stage and old_stage and stage == old_stage and next_progress < old_progress:
                next_progress = old_progress
            job["progress"] = next_progress

        if message:
            job["message"] = message
            if message != job.get("_last_message"):
                logs = job.get("logs", [])
                logs.append(message)
                if len(logs) > 300:
                    logs = logs[-300:]
                job["logs"] = logs
                job["_last_message"] = message


def _parse_progress_text(job_id: str, text: str):
    if not text:
        return

    clean = " ".join(text.replace("\t", " ").strip().split())
    if not clean:
        return

    shard_match = _SHARD_RE.search(clean)
    if shard_match:
        current = int(shard_match.group(1))
        total = int(shard_match.group(2))
        pct = int((current / total) * 100) if total else 0
        _update_load_progress(
            job_id,
            stage="loading_shards",
            current=current,
            total=total,
            progress=pct,
            message=f"Loading shard {current}/{total}",
        )
        return

    layers_match = _LAYERS_RE.search(clean)
    if layers_match:
        current = int(layers_match.group(1))
        total = int(layers_match.group(2))
        pct_match = _PERCENT_RE.search(clean)
        pct = int(pct_match.group(1)) if pct_match else (int((current / total) * 100) if total else 0)
        _update_load_progress(
            job_id,
            stage="loading_layers",
            current=current,
            total=total,
            progress=pct,
            message=f"Loading layers {current}/{total}",
        )
        return

    count_match = _COUNT_RE.search(clean)
    pct_match = _PERCENT_RE.search(clean)
    if count_match and ("layer" in clean.lower() or "shard" in clean.lower()):
        current = int(count_match.group(1))
        total = int(count_match.group(2))
        pct = int(pct_match.group(1)) if pct_match else (int((current / total) * 100) if total else 0)
        label = "Loading shards" if "shard" in clean.lower() else "Loading layers"
        _update_load_progress(
            job_id,
            stage="loading_shards" if "shard" in clean.lower() else "loading_layers",
            current=current,
            total=total,
            progress=pct,
            message=f"{label} {current}/{total}",
        )
        return

    if pct_match:
        pct = int(pct_match.group(1))
        _update_load_progress(
            job_id,
            stage="loading",
            progress=pct,
            message=clean,
        )
        return

    if any(keyword in clean.lower() for keyword in ("loading", "tokenizer", "config", "split", "checkpoint")):
        _update_load_progress(job_id, stage="loading", message=clean)


class _LoadProgressStream(io.TextIOBase):
    def __init__(self, job_id: str, passthrough):
        super().__init__()
        self._job_id = job_id
        self._passthrough = passthrough

    def write(self, s):
        if s is None:
            return 0
        text = s if isinstance(s, str) else str(s)

        if self._passthrough is not None:
            try:
                self._passthrough.write(text)
                self._passthrough.flush()
            except Exception:
                pass

        for chunk in re.split(r"[\r\n]+", text):
            _parse_progress_text(self._job_id, chunk)

        return len(text)

    def flush(self):
        if self._passthrough is not None:
            try:
                self._passthrough.flush()
            except Exception:
                pass


def _run_load_job(job_id: str, payload: dict):
    _load_job_update(
        job_id,
        status="loading",
        stage="starting",
        message="Starting model load...",
        progress=0,
    )
    _load_job_log(job_id, "Starting model load.")

    out_stream = _LoadProgressStream(job_id, sys.stdout)
    err_stream = _LoadProgressStream(job_id, sys.stderr)

    try:
        with redirect_stdout(out_stream), redirect_stderr(err_stream):
            load_model(
                model_id=payload.get("model_id"),
                model_path=payload.get("model_path"),
                model_base_dir=payload.get("model_base_dir"),
                device=payload.get("device"),
                force_reload=bool(payload.get("force_reload", False)),
            )

        if bool(payload.get("persist_env", False)):
            persist_runtime_to_env()
            _load_job_log(job_id, f"Persisted settings to {ENV_FILE}")

        snapshot = runtime_state()
        _load_job_update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Model loaded",
            error=None,
            finished_at=time.time(),
            result=snapshot,
        )
        _load_job_log(job_id, "Model load completed.")
    except Exception as exc:
        _load_job_update(
            job_id,
            status="failed",
            stage="failed",
            message=f"Model load failed: {exc}",
            error=str(exc),
            finished_at=time.time(),
        )
        _load_job_log(job_id, f"Model load failed: {exc}")


def start_load_job(payload: dict):
    with _load_jobs_lock:
        active = [job for job in _load_jobs.values() if job.get("status") in ("queued", "loading")]
        if active:
            active.sort(key=lambda item: item.get("started_at", 0), reverse=True)
            return active[0].get("job_id")

    job_id = _new_load_job(payload)
    thread = threading.Thread(target=_run_load_job, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id