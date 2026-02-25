import io
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import torch
from dotenv import set_key

# Always prioritize the local bundled air_llm package so runtime behavior
# matches repository code changes and does not depend on stale site-packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_AIR_LLM_ROOT = _PROJECT_ROOT / "air_llm"
if _LOCAL_AIR_LLM_ROOT.exists():
    local_pkg_path = str(_LOCAL_AIR_LLM_ROOT)
    if local_pkg_path not in sys.path:
        sys.path.insert(0, local_pkg_path)

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


def _looks_like_local_path(value: Optional[str]) -> bool:
    value = normalize_optional(value)
    if not value:
        return False
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        return True
    if value.startswith(("\\\\", "./", ".\\", "../", "..\\", "/", "~")):
        return True
    if "\\" in value:
        return True
    return os.path.isabs(os.path.expanduser(value))


def looks_like_local_path(value: Optional[str]) -> bool:
    return _looks_like_local_path(value)


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


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_safetensors_index(local_dir: Path) -> Optional[Path]:
    safetensor_files = sorted(local_dir.glob("*.safetensors"))
    if not safetensor_files:
        return None

    try:
        from safetensors import safe_open
    except Exception:
        return None

    weight_map = {}
    total_size = 0
    for file_path in safetensor_files:
        total_size += file_path.stat().st_size
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = file_path.name

    if not weight_map:
        return None

    index_path = local_dir / "model.safetensors.index.json"
    _write_json(index_path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    return index_path


def _build_pytorch_bin_index(local_dir: Path) -> Optional[Path]:
    bin_path = local_dir / "pytorch_model.bin"
    if not bin_path.exists() or not bin_path.is_file():
        return None

    try:
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(str(bin_path), map_location="cpu")
    except Exception:
        return None

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict) or not state_dict:
        return None

    weight_map = {str(key): "pytorch_model.bin" for key in state_dict.keys()}
    index_path = local_dir / "pytorch_model.bin.index.json"
    _write_json(index_path, {"metadata": {"total_size": bin_path.stat().st_size}, "weight_map": weight_map})
    return index_path


def _ensure_airllm_index_files(local_source: str):
    local_dir = Path(local_source)
    if not local_dir.is_dir():
        return

    has_index = (
        (local_dir / "model.safetensors.index.json").exists()
        or (local_dir / "pytorch_model.bin.index.json").exists()
    )
    if has_index:
        return

    generated = _build_safetensors_index(local_dir)
    if generated is not None:
        print(f"Generated {generated.name} for local model loading.")
        return

    generated = _build_pytorch_bin_index(local_dir)
    if generated is not None:
        print(f"Generated {generated.name} for local model loading.")
        return


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
        local_expected = bool(
            target_model_path
            or target_model_base_dir
            or _looks_like_local_path(target_model_source)
        )
        source_for_load = target_model_source
        if local_expected:
            local_source = os.path.expanduser(target_model_source)
            if not os.path.exists(local_source):
                raise FileNotFoundError(f"Local model path does not exist: {target_model_source}")
            _ensure_airllm_index_files(local_source)
            validate_local_model_source(local_source)
            source_for_load = local_source

        _model = AutoModel.from_pretrained(source_for_load, **kwargs)
        _runtime.update(
            {
                "model_id": target_model_id,
                "model_path": target_model_path,
                "model_base_dir": target_model_base_dir,
                "device": target_device,
                "model_source": source_for_load,
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

