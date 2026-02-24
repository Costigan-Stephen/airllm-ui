import json
import os
import threading
import time
import uuid
from html import escape
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from dotenv import load_dotenv, set_key
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from tqdm.auto import tqdm as tqdm_auto

from airllm import AutoModel


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=False)


def _find_favicon():
    for name in ("favicon.ico", "favicon.png", "favicon.svg", "favicon.jpg", "favicon.jpeg"):
        candidate = PROJECT_ROOT / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


FAVICON_PATH = _find_favicon()


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _getenv_int(name: str, default: int) -> int:
    value = _getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


MODEL_ID_DEFAULT = _getenv("AIRLLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODEL_PATH_DEFAULT = _getenv("AIRLLM_MODEL_PATH")
MODEL_BASE_DIR_DEFAULT = _getenv("AIRLLM_MODEL_BASE_DIR")
DEVICE_DEFAULT = _getenv("AIRLLM_DEVICE", _default_device())
if DEVICE_DEFAULT:
    DEVICE_DEFAULT = DEVICE_DEFAULT.split()[0]
    if DEVICE_DEFAULT == "cuda":
        DEVICE_DEFAULT = "cuda:0"
HF_TOKEN = _getenv("HF_TOKEN")
MAX_INPUT_TOKENS = _getenv_int("AIRLLM_MAX_INPUT_TOKENS", 1024)
PORT = _getenv_int("PORT", 8000)

_model = None
_model_lock = threading.Lock()
_download_jobs = {}
_download_lock = threading.Lock()
_runtime = {
    "model_id": MODEL_ID_DEFAULT,
    "model_path": MODEL_PATH_DEFAULT,
    "model_base_dir": MODEL_BASE_DIR_DEFAULT,
    "device": DEVICE_DEFAULT,
    "model_source": None,
}

app = FastAPI(title="AirLLM API", version="1.0.0")


def _normalize_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value if value else None


def _normalize_device(value: Optional[str]) -> Optional[str]:
    value = _normalize_optional(value)
    if not value:
        return None
    value = value.split()[0]
    if value == "cuda":
        return "cuda:0"
    return value


def _resolve_model_source(
    model_id: str,
    model_path: Optional[str],
    model_base_dir: Optional[str],
) -> str:
    model_path = _normalize_optional(model_path)
    model_base_dir = _normalize_optional(model_base_dir)
    if model_path:
        return model_path
    if model_base_dir:
        return os.path.join(model_base_dir, *model_id.split("/"))
    return model_id


def _runtime_state():
    return {
        "model_loaded": _model is not None,
        "model_id": _runtime["model_id"],
        "model_source": _runtime["model_source"]
        or _resolve_model_source(
            _runtime["model_id"],
            _runtime["model_path"],
            _runtime["model_base_dir"],
        ),
        "model_path": _runtime["model_path"],
        "model_base_dir": _runtime["model_base_dir"],
        "device": _runtime["device"],
    }


def _persist_runtime_to_env():
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), "AIRLLM_MODEL_ID", _runtime["model_id"] or "", quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_PATH", _runtime["model_path"] or "", quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_BASE_DIR", _runtime["model_base_dir"] or "", quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_DEVICE", _runtime["device"] or "", quote_mode="never")


def _new_download_job(model_id: str, base_dir: str, target_dir: str):
    job_id = str(uuid.uuid4())
    with _download_lock:
        _download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model_id": model_id,
            "base_dir": base_dir,
            "target_dir": target_dir,
            "progress": 0,
            "current": 0,
            "total": None,
            "logs": [f"Job created for {model_id}"],
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
    return job_id


def _job_update(job_id: str, **fields):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def _job_log(job_id: str, message: str):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return
        logs = job.get("logs", [])
        logs.append(message)
        # Keep payload bounded for UI polling.
        if len(logs) > 300:
            logs = logs[-300:]
        job["logs"] = logs


def _job_snapshot(job_id: str):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return None
        return dict(job)


def _inspect_local_model_dir(root_path: Path, files, exhaustive: bool = False):
    file_set = set(files)
    has_gguf = any(name.lower().endswith(".gguf") for name in files)
    has_config = "config.json" in file_set
    has_weights = (
        "pytorch_model.bin" in file_set
        or "pytorch_model.bin.index.json" in file_set
        or any(name.endswith(".safetensors") for name in files)
    )

    model_type = None
    config_error = None
    if has_config:
        config_path = root_path / "config.json"
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = config_data.get("model_type")
        except Exception as exc:
            config_error = f"Invalid config.json: {exc}"

    if has_gguf and not (has_config and has_weights and model_type):
        return False, "GGUF directory (not a Transformers checkpoint for AirLLM)."
    if has_config and config_error:
        return False, config_error
    if has_config and not model_type:
        return False, "config.json exists but has no model_type."
    if has_config and model_type and not has_weights:
        return False, f"model_type={model_type} but no PyTorch/safetensors weights found."
    if has_config and model_type and has_weights:
        return True, f"model_type={model_type}"
    if exhaustive:
        return False, "No compatible Transformers config.json + weights found."
    return False, None


def _validate_local_model_source(model_source: str):
    source_path = Path(model_source)
    if source_path.is_file() and source_path.suffix.lower() == ".gguf":
        raise ValueError(
            f"Selected model is GGUF ({model_source}). AirLLM expects a Hugging Face Transformers checkpoint directory."
        )
    if not source_path.is_dir():
        return

    files = [entry.name for entry in source_path.iterdir() if entry.is_file()]
    compatible, reason = _inspect_local_model_dir(source_path, files, exhaustive=True)
    if compatible:
        return

    if reason and "GGUF" in reason:
        raise ValueError(
            f"Selected model path is GGUF-based: {model_source}. Choose a Transformers model folder with config.json + model weights."
        )
    if reason and "no model_type" in reason:
        raise ValueError(
            f"Selected model path is not loadable by Transformers: {model_source} ({reason})"
        )
    if reason:
        raise ValueError(
            f"Selected local model path is not compatible with AirLLM: {model_source} ({reason})"
        )


def _discover_models(base_dir: str, max_depth: int = 4, limit: int = 500):
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return [], [], [f"Base directory not found: {base_dir}"]

    models = []
    unsupported_models = []
    errors = []

    def _onerror(err):
        errors.append(str(err))

    for root, dirs, files in os.walk(base_path, onerror=_onerror):
        root_path = Path(root)
        rel_path = root_path.relative_to(base_path)
        depth = len(rel_path.parts)
        if depth > max_depth:
            dirs[:] = []
            continue

        compatible, reason = _inspect_local_model_dir(root_path, files)
        if compatible:
            label = "." if str(rel_path) == "." else str(rel_path).replace("\\", "/")
            models.append({"label": label, "path": str(root_path)})
            dirs[:] = []
        elif reason:
            label = "." if str(rel_path) == "." else str(rel_path).replace("\\", "/")
            unsupported_models.append({"label": label, "path": str(root_path), "reason": reason})

        if len(models) >= limit:
            break

    models.sort(key=lambda item: item["label"].lower())
    unsupported_models.sort(key=lambda item: item["label"].lower())
    return models, unsupported_models, errors


def _list_candidate_directories(base_dir: str, max_depth: int = 2, limit: int = 300):
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return []

    directories = []
    for root, dirs, _ in os.walk(base_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(base_path)
        depth = len(rel_path.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if depth > 0:
            label = str(rel_path).replace("\\", "/")
            directories.append({"label": label, "path": str(root_path)})
        if len(directories) >= limit:
            break

    directories.sort(key=lambda item: item["label"].lower())
    return directories


def _parse_patterns(patterns: Optional[str]):
    patterns = _normalize_optional(patterns)
    if not patterns:
        return None
    values = []
    for chunk in patterns.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values or None


def _resolve_hf_download_dir(base_dir: str, model_id: str, subdir: Optional[str]):
    model_id = model_id.strip()
    base = Path(base_dir)
    if subdir:
        rel = Path(subdir.strip())
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("subdir must be a relative path under the base directory.")
        return (base / rel).resolve()
    return (base / Path(*model_id.split("/"))).resolve()


def _run_hf_download_job(job_id: str, payload: dict):
    global _model

    model_id = payload["model_id"]
    base_dir = payload["base_dir"]
    target_dir = payload["target_dir"]
    revision = payload.get("revision")
    allow_patterns = payload.get("allow_patterns")
    ignore_patterns = payload.get("ignore_patterns")
    set_as_active_model = payload.get("set_as_active_model", True)
    persist_env = payload.get("persist_env", True)

    _job_update(job_id, status="downloading")
    _job_log(job_id, f"Starting download for {model_id}")
    _job_log(job_id, f"Target directory: {target_dir}")

    class JobTqdm(tqdm_auto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _job_update(job_id, total=self.total or None)
            self._last_emit = 0.0

        def update(self, n=1):
            out = super().update(n)
            now = time.time()
            current = int(self.n or 0)
            total = int(self.total) if self.total else None
            progress = int((current / total) * 100) if total else 0
            _job_update(job_id, current=current, total=total, progress=progress)
            if now - self._last_emit > 0.5:
                _job_log(job_id, f"Download progress: {current}/{total if total else '?'} files")
                self._last_emit = now
            return out

    try:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=target_dir,
            token=HF_TOKEN,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            tqdm_class=JobTqdm,
        )
    except Exception as exc:
        message = f"Hugging Face download failed: {exc}"
        _job_log(job_id, message)
        _job_update(job_id, status="failed", error=str(exc), finished_at=time.time())
        return

    _job_log(job_id, "Download finished successfully.")

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

    if persist_env:
        _persist_runtime_to_env()
        _job_log(job_id, f"Persisted settings to {ENV_FILE}")

    _job_update(
        job_id,
        status="completed",
        progress=100,
        error=None,
        finished_at=time.time(),
        set_as_active_model=set_as_active_model,
        persist_env=persist_env,
    )


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(64, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=500)


class LoadRequest(BaseModel):
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_base_dir: Optional[str] = None
    device: Optional[str] = None
    force_reload: bool = False
    persist_env: bool = False


class SettingsRequest(BaseModel):
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_base_dir: Optional[str] = None
    device: Optional[str] = None
    persist_env: bool = True


class HFDownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    base_dir: Optional[str] = None
    subdir: Optional[str] = None
    revision: Optional[str] = None
    allow_patterns: Optional[str] = None
    ignore_patterns: Optional[str] = None
    set_as_active_model: bool = True
    persist_env: bool = True


class GenerateResponse(BaseModel):
    model_id: str
    model_source: str
    device: str
    text: str
    generated_text: str


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
        target_model_id = _normalize_optional(model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
        target_model_path = _normalize_optional(model_path)
        if target_model_path is None:
            target_model_path = _runtime["model_path"]
        target_model_base_dir = _normalize_optional(model_base_dir)
        if target_model_base_dir is None:
            target_model_base_dir = _runtime["model_base_dir"]
        target_device = _normalize_device(device) or _runtime["device"] or DEVICE_DEFAULT
        target_model_source = _resolve_model_source(
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
            _validate_local_model_source(target_model_source)
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


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AirLLM UI</title>
  <link rel="icon" href="/favicon.ico" type="image/x-icon">
  <style>
    :root {
      --bg: #f5f7fb;
      --bg-soft: #eef2ff;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #0f172a;
      --accent-soft: #e2e8f0;
      --accent-text: #ffffff;
      --shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
      --message-user: #dbeafe;
      --message-user-border: #bfdbfe;
      --message-assistant: #f8fafc;
      --message-assistant-border: #e2e8f0;
    }
    [data-theme="dark"] {
      --bg: #0b1220;
      --bg-soft: #131c2f;
      --panel: #111827;
      --panel-soft: #0f172a;
      --border: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #2563eb;
      --accent-soft: #1e293b;
      --accent-text: #f8fafc;
      --shadow: 0 10px 30px rgba(2, 6, 23, 0.45);
      --message-user: #1e3a8a;
      --message-user-border: #1d4ed8;
      --message-assistant: #111827;
      --message-assistant-border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at 15% 0%, var(--bg-soft), transparent 40%),
        radial-gradient(circle at 90% 10%, var(--accent-soft), transparent 30%),
        var(--bg);
      color: var(--text);
      transition: background 0.2s ease, color 0.2s ease;
    }
    .shell {
      display: grid;
      grid-template-columns: 260px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      backdrop-filter: blur(8px);
      padding: 16px;
    }
    .brand {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 16px;
    }
    .tab-btn {
      width: 100%;
      text-align: left;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 8px;
      cursor: pointer;
      transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
    }
    .tab-btn:hover {
      transform: translateY(-1px);
      border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
    }
    .tab-btn.active {
      background: var(--accent);
      color: var(--accent-text);
      border-color: var(--accent);
    }
    .meta {
      margin-top: 16px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      font-size: 12px;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .main {
      display: grid;
      grid-template-rows: 56px 1fr;
      min-width: 0;
    }
    .topbar {
      border-bottom: 1px solid var(--border);
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      gap: 8px;
    }
    .topbar .title {
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .topbar-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .topbar-model {
      max-width: 360px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .icon-btn {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      cursor: pointer;
    }
    .content {
      padding: 16px;
      min-height: 0;
    }
    .panel {
      display: none;
      height: 100%;
    }
    .panel.active {
      display: block;
    }
    .chat-layout {
      display: grid;
      grid-template-rows: 1fr auto;
      height: calc(100vh - 88px);
      gap: 12px;
    }
    #messages {
      background: color-mix(in srgb, var(--panel-soft) 65%, var(--panel));
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: auto;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .message {
      padding: 11px 13px;
      border-radius: 14px;
      margin-bottom: 12px;
      max-width: 90%;
      white-space: pre-wrap;
      line-height: 1.45;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
    .message.user {
      margin-left: auto;
      background: var(--message-user);
      border: 1px solid var(--message-user-border);
    }
    .message.assistant {
      margin-right: auto;
      background: var(--message-assistant);
      border: 1px solid var(--message-assistant-border);
    }
    .composer {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      display: grid;
      gap: 8px;
      box-shadow: var(--shadow);
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
      background: var(--panel);
      color: var(--text);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    textarea:focus, input:focus, select:focus {
      outline: none;
      border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
    }
    textarea { resize: vertical; min-height: 78px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      margin-bottom: 12px;
      box-shadow: var(--shadow);
    }
    .card h3 { margin: 0 0 8px; font-size: 15px; }
    .label { font-size: 12px; font-weight: 600; color: var(--muted); margin-top: 10px; display: block; }
    .btn {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }
    .btn:hover {
      transform: translateY(-1px);
      border-color: color-mix(in srgb, var(--accent) 40%, var(--border));
    }
    .btn.primary {
      background: var(--accent);
      color: var(--accent-text);
      border-color: var(--accent);
    }
    .inline { display: inline-flex; align-items: center; gap: 8px; }
    .small { font-size: 12px; color: var(--muted); }
    #settingsStatus { white-space: pre-wrap; background: #f9fafb; border: 1px solid var(--border); border-radius: 10px; padding: 10px; max-height: 220px; overflow: auto; }
    .settings-tabs { display: flex; gap: 8px; margin-bottom: 12px; }
    .settings-tab-btn {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .settings-tab-btn.active {
      background: var(--accent);
      color: var(--accent-text);
      border-color: var(--accent);
    }
    .settings-subpanel { display: none; }
    .settings-subpanel.active { display: block; }
    .download-badge {
      font-size: 10px;
      border-radius: 999px;
      background: #f59e0b;
      color: #111827;
      padding: 2px 7px;
      font-weight: 700;
    }
    .hidden { display: none !important; }
    .progress-wrap {
      margin-top: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: var(--panel);
    }
    .progress-bar {
      height: 10px;
      width: 0%;
      background: #0f766e;
      transition: width 0.2s ease;
    }
    .log-box {
      margin-top: 8px;
      max-height: 180px;
      overflow: auto;
      white-space: pre-wrap;
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.55);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 30;
      padding: 16px;
    }
    .modal-overlay.active { display: flex; }
    .modal-card {
      width: min(900px, 96vw);
      max-height: 90vh;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .modal-title {
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 700;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 12px;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
      .chat-layout { height: auto; min-height: 70vh; }
      .row, .row3 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">AirLLM UI</div>
      <button id="tabChatBtn" class="tab-btn active" onclick="switchTab('chat')">Chat</button>
      <button id="tabSettingsBtn" class="tab-btn" onclick="switchTab('settings')">Settings</button>
      <div id="runtimeMeta" class="meta">Loading...</div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="title" id="topbarTitle">Chat</div>
        <div class="topbar-actions">
          <div class="small topbar-model" id="topbarModel">Model: -</div>
          <button id="themeToggle" class="icon-btn" onclick="toggleTheme()">Dark</button>
        </div>
      </div>

      <div class="content">
        <section id="chatPanel" class="panel active">
          <div class="chat-layout">
            <div id="messages">
              <div class="message assistant">AirLLM is ready. Open Settings to choose model source, then send a prompt.</div>
            </div>
            <div class="composer">
              <textarea id="prompt" placeholder="Type a message. Enter to send, Shift+Enter for a new line."></textarea>
              <div class="row">
                <div>
                  <span class="label">Max New Tokens</span>
                  <input id="maxNewTokens" type="number" value="64" min="1" max="1024" />
                </div>
                <div>
                  <span class="label">Do Sample (true/false)</span>
                  <input id="doSample" value="false" />
                </div>
              </div>
              <div style="display:flex; justify-content:flex-end;">
                <span class="small" style="margin-right:auto; align-self:center;">Press Enter to send</span>
                <button class="btn primary" onclick="sendPrompt()">Send</button>
              </div>
            </div>
          </div>
        </section>

        <section id="settingsPanel" class="panel">
          <div class="settings-tabs">
            <button id="settingsTabConfigBtn" class="settings-tab-btn active" onclick="switchSettingsTab('config')">Model Config</button>
            <button id="settingsTabDownloadBtn" class="settings-tab-btn" onclick="switchSettingsTab('download')">Hugging Face Download <span id="settingsDownloadBadge" class="download-badge hidden">Downloading</span></button>
          </div>

          <div id="settingsConfigPanel" class="settings-subpanel active">
            <div class="card">
              <h3>Model Configuration</h3>
              <span class="label">Model ID (HF repo id)</span>
              <input id="modelId" value="__MODEL_ID_VALUE__" placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0" />
              <span class="label">External Model Path (AIRLLM_MODEL_PATH)</span>
              <input id="modelPath" value="__MODEL_PATH_VALUE__" placeholder="Path to HF Transformers model directory" />
              <span class="label">External Base Directory (AIRLLM_MODEL_BASE_DIR)</span>
              <input id="modelBaseDir" value="__MODEL_BASE_DIR_VALUE__" placeholder="AIRLLM_MODEL_BASE_DIR from .env" />
              <div class="row3">
                <div>
                  <span class="label">Discovered Models</span>
                  <select id="modelSelect" onchange="applySelectedModel()">
                    <option value="">(scan base directory)</option>
                  </select>
                  <div id="modelScanInfo" class="small" style="margin-top:6px;">Scan to list compatible Transformers model folders.</div>
                </div>
                <div>
                  <span class="label">Actions</span>
                  <button class="btn" onclick="refreshModels()">Scan Base Directory</button>
                </div>
                <div>
                  <span class="label">Actions</span>
                  <button class="btn" onclick="clearModelPath()">Clear Model Path</button>
                </div>
              </div>

              <span class="label">Device</span>
              <input id="device" value="__DEVICE_VALUE__" placeholder="cuda:0 or cpu" />
              <div class="small">Recommended for this machine: <code>cuda:0</code>. Use <code>cpu</code> only if needed.</div>
              <div class="small">Examples: <code>cuda:0</code>, <code>cpu</code>. Value must not include extra text.</div>
              <div class="inline" style="margin-top:10px;">
                <input id="persistEnv" type="checkbox" checked style="width:auto;" />
                <span class="small">Persist selected values to project <code>.env</code></span>
              </div>
              <div style="display:flex; gap:8px; margin-top:12px;">
                <button class="btn" onclick="saveSettings()">Save Settings</button>
                <button class="btn primary" onclick="loadModel()">Load / Reload Model</button>
              </div>
            </div>
          </div>

          <div id="settingsDownloadPanel" class="settings-subpanel">
            <div class="card">
              <h3>Hugging Face Download</h3>
              <span class="label">Repo ID</span>
              <input id="hfModelId" placeholder="meta-llama/Llama-2-7b-hf" />
              <div class="row">
                <div>
                  <span class="label">Revision (optional)</span>
                  <input id="hfRevision" placeholder="main" />
                </div>
                <div>
                  <span class="label">Target Subdir (optional)</span>
                  <input id="hfSubdir" placeholder="custom/my-model-dir" />
                </div>
              </div>
              <span class="label">Allow Patterns (optional, comma separated)</span>
              <input id="hfAllowPatterns" placeholder="config.json,*.safetensors,tokenizer*" />
              <span class="label">Ignore Patterns (optional, comma separated)</span>
              <input id="hfIgnorePatterns" placeholder="*.onnx,*.msgpack" />
              <div class="inline" style="margin-top:10px;">
                <input id="hfSetActive" type="checkbox" checked style="width:auto;" />
                <span class="small">Set downloaded model as active AIRLLM_MODEL_PATH</span>
              </div>
              <div style="display:flex; gap:8px; margin-top:12px;">
                <button class="btn primary" onclick="downloadFromHF()">Download from Hugging Face</button>
              </div>
              <div class="small" style="margin-top:8px;">Downloads are saved under the current base directory.</div>
            </div>

            <div class="card">
              <h3>Download Progress</h3>
              <div id="downloadJobMeta" class="small">No active downloads.</div>
              <div class="progress-wrap"><div id="downloadProgressBar" class="progress-bar"></div></div>
              <div id="downloadProgressText" class="small" style="margin-top:6px;">0%</div>
              <div id="downloadLogBox" class="log-box">No logs yet.</div>
            </div>
          </div>

          <div class="card">
            <h3>Settings Status</h3>
            <div id="settingsStatus">No actions yet.</div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <div id="downloadModal" class="modal-overlay">
    <div class="modal-card">
      <h3 class="modal-title">Model Not Found Locally</h3>
      <div class="small">The selected model path does not exist on disk. Download it from Hugging Face now?</div>
      <div id="downloadModalHint" class="small" style="margin-top:8px;"></div>

      <span class="label">Repo ID</span>
      <input id="modalHfModelId" placeholder="meta-llama/Llama-2-7b-hf" />
      <div class="row">
        <div>
          <span class="label">Revision (optional)</span>
          <input id="modalHfRevision" placeholder="main" />
        </div>
        <div>
          <span class="label">Target Subdir (optional)</span>
          <input id="modalHfSubdir" placeholder="custom/my-model-dir" />
        </div>
      </div>
      <span class="label">Allow Patterns (optional, comma separated)</span>
      <input id="modalHfAllowPatterns" placeholder="config.json,*.safetensors,tokenizer*" />
      <span class="label">Ignore Patterns (optional, comma separated)</span>
      <input id="modalHfIgnorePatterns" placeholder="*.onnx,*.msgpack" />
      <div class="inline" style="margin-top:10px;">
        <input id="modalHfSetActive" type="checkbox" checked style="width:auto;" />
        <span class="small">Set downloaded model as active model path</span>
      </div>

      <div class="modal-actions">
        <button class="btn" onclick="closeDownloadModal()">Cancel</button>
        <button class="btn primary" onclick="downloadFromHFModal()">Download</button>
      </div>
    </div>
  </div>

  <script>
    let lastStatus = null;
    let currentTheme = "light";
    let isGenerating = false;
    let activeDownloadJobId = null;
    let downloadPollTimer = null;
    let pendingLoadPayload = null;

    function switchTab(tab) {
      const isChat = tab === "chat";
      document.getElementById("chatPanel").classList.toggle("active", isChat);
      document.getElementById("settingsPanel").classList.toggle("active", !isChat);
      document.getElementById("tabChatBtn").classList.toggle("active", isChat);
      document.getElementById("tabSettingsBtn").classList.toggle("active", !isChat);
      document.getElementById("topbarTitle").textContent = isChat ? "Chat" : "Settings";
    }

    function setDownloadTabIndicator(active) {
      const badge = document.getElementById("settingsDownloadBadge");
      if (!badge) return;
      badge.classList.toggle("hidden", !active);
    }

    function applyTheme(theme) {
      currentTheme = theme === "dark" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", currentTheme);
      localStorage.setItem("airllm_theme", currentTheme);
      const toggle = document.getElementById("themeToggle");
      if (toggle) {
        toggle.textContent = currentTheme === "dark" ? "Light" : "Dark";
      }
    }

    function toggleTheme() {
      applyTheme(currentTheme === "dark" ? "light" : "dark");
    }

    function initTheme() {
      const stored = localStorage.getItem("airllm_theme");
      if (stored === "dark" || stored === "light") {
        applyTheme(stored);
        return;
      }
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      applyTheme(prefersDark ? "dark" : "light");
    }

    function switchSettingsTab(tab) {
      const isConfig = tab === "config";
      document.getElementById("settingsConfigPanel").classList.toggle("active", isConfig);
      document.getElementById("settingsDownloadPanel").classList.toggle("active", !isConfig);
      document.getElementById("settingsTabConfigBtn").classList.toggle("active", isConfig);
      document.getElementById("settingsTabDownloadBtn").classList.toggle("active", !isConfig);
    }

    function modelDisplayName(data) {
      if (data && data.model_id) return data.model_id;
      const src = (data && data.model_source) || "";
      if (!src) return "-";
      const parts = src.split(/[\\\\/]/).filter(Boolean);
      return parts.length ? parts[parts.length - 1] : src;
    }

    function setSettingsStatus(obj) {
      document.getElementById("settingsStatus").textContent = JSON.stringify(obj, null, 2);
    }

    function appendMessage(role, text) {
      const box = document.getElementById("messages");
      const el = document.createElement("div");
      el.className = "message " + role;
      el.textContent = text;
      box.appendChild(el);
      box.scrollTop = box.scrollHeight;
    }

    function asBool(value) {
      return String(value).toLowerCase() === "true";
    }

    async function apiJson(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) {
        if (data && typeof data === "object" && data.detail) {
          throw new Error(String(data.detail));
        }
        throw new Error(typeof data === "object" ? JSON.stringify(data) : String(data));
      }
      return data;
    }

    function syncModalFieldsFromSettings() {
      document.getElementById("modalHfModelId").value = document.getElementById("hfModelId").value || document.getElementById("modelId").value || "";
      document.getElementById("modalHfRevision").value = document.getElementById("hfRevision").value || "";
      document.getElementById("modalHfSubdir").value = document.getElementById("hfSubdir").value || "";
      document.getElementById("modalHfAllowPatterns").value = document.getElementById("hfAllowPatterns").value || "";
      document.getElementById("modalHfIgnorePatterns").value = document.getElementById("hfIgnorePatterns").value || "";
      document.getElementById("modalHfSetActive").checked = document.getElementById("hfSetActive").checked;
    }

    function openDownloadModal(resolveData) {
      syncModalFieldsFromSettings();
      if (resolveData && resolveData.model_id) {
        document.getElementById("modalHfModelId").value = resolveData.model_id;
      }
      const modal = document.getElementById("downloadModal");
      document.getElementById("downloadModalHint").textContent =
        "Expected local path: " + (resolveData.model_source || "(unknown)") + " | Base directory: " + (resolveData.model_base_dir || "(unset)");
      modal.classList.add("active");
    }

    function closeDownloadModal(clearPending = true) {
      document.getElementById("downloadModal").classList.remove("active");
      if (clearPending) {
        pendingLoadPayload = null;
      }
    }

    function setDownloadProgress(job) {
      const bar = document.getElementById("downloadProgressBar");
      const pct = Number(job.progress || 0);
      bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
      document.getElementById("downloadProgressText").textContent =
        (job.status || "unknown") + " - " + pct + "% (" + (job.current || 0) + "/" + (job.total || "?") + ")";
      document.getElementById("downloadJobMeta").textContent =
        "Job: " + (job.job_id || "-") + " | Model: " + (job.model_id || "-") + " | Target: " + (job.target_dir || "-");
      const logs = job.logs || [];
      document.getElementById("downloadLogBox").textContent = logs.length ? logs.join("\\n") : "No logs yet.";
      setDownloadTabIndicator(job.status === "queued" || job.status === "downloading");
    }

    function stopDownloadPolling() {
      if (downloadPollTimer) {
        clearInterval(downloadPollTimer);
        downloadPollTimer = null;
      }
      activeDownloadJobId = null;
      localStorage.removeItem("airllm_download_job_id");
      setDownloadTabIndicator(false);
    }

    async function pollDownloadJob(jobId, options = {}) {
      const closeModalOnComplete = !!options.closeModalOnComplete;
      const triggerPendingLoad = !!options.triggerPendingLoad;
      stopDownloadPolling();
      activeDownloadJobId = jobId;
      localStorage.setItem("airllm_download_job_id", jobId);
      setDownloadTabIndicator(true);

      async function tick() {
        if (!activeDownloadJobId) return;
        const data = await apiJson("/hf/download/" + encodeURIComponent(activeDownloadJobId));
        setDownloadProgress(data);
        if (data.status === "failed") {
          stopDownloadPolling();
          setSettingsStatus(data);
          appendMessage("assistant", "Download failed: " + (data.error || "unknown error"));
          if (closeModalOnComplete) {
            closeDownloadModal();
          }
          return;
        }
        if (data.status === "completed") {
          stopDownloadPolling();
          setSettingsStatus(data);
          appendMessage("assistant", "Download completed: " + (data.model_id || ""));
          await refreshStatus();
          await refreshModels(document.getElementById("modelPath").value || "", true);
          if (closeModalOnComplete) {
            closeDownloadModal(false);
          }
          if (triggerPendingLoad && pendingLoadPayload) {
            const payload = pendingLoadPayload;
            pendingLoadPayload = null;
            await executeLoad(payload);
          }
        }
      }

      await tick();
      if (activeDownloadJobId) {
        downloadPollTimer = setInterval(() => {
          tick().catch((err) => {
            stopDownloadPolling();
            setSettingsStatus({ error: String(err) });
          });
        }, 1000);
      }
    }

    function applySelectedModel() {
      const selected = document.getElementById("modelSelect").value;
      if (selected) {
        document.getElementById("modelPath").value = selected;
      }
    }

    function clearModelPath() {
      document.getElementById("modelPath").value = "";
      document.getElementById("modelSelect").value = "";
    }

    async function refreshModels(selectedPath = "", keepStatus = false) {
      const baseDir = document.getElementById("modelBaseDir").value || "";
      const select = document.getElementById("modelSelect");
      const info = document.getElementById("modelScanInfo");
      select.innerHTML = "";
      if (!baseDir) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "(set AIRLLM_MODEL_BASE_DIR first)";
        select.appendChild(option);
        info.textContent = "Set a base directory, then scan.";
        return;
      }

      const data = await apiJson("/models?base_dir=" + encodeURIComponent(baseDir));

      const empty = document.createElement("option");
      empty.value = "";
      const modelCount = (data.models || []).length;
      empty.textContent = modelCount ? "(select compatible model directory)" : "(no compatible Transformers models found)";
      select.appendChild(empty);

      for (const model of (data.models || [])) {
        const option = document.createElement("option");
        option.value = model.path;
        option.textContent = "[model] " + model.label + " -> " + model.path;
        if (selectedPath && selectedPath === model.path) {
          option.selected = true;
        }
        select.appendChild(option);
      }

      if (selectedPath) {
        const existsInList = (data.models || []).some((m) => m.path === selectedPath);
        if (!existsInList) {
          const current = document.createElement("option");
          current.value = selectedPath;
          current.selected = true;
          current.textContent = "[current/manual] " + selectedPath;
          select.appendChild(current);
        }
      }

      const unsupportedCount = (data.unsupported_models || []).length;
      const scanErrorCount = (data.scan_errors || []).length;
      info.textContent =
        "Compatible models: " + modelCount +
        " | Unsupported detected: " + unsupportedCount +
        " | Scan errors: " + scanErrorCount;

      if (!keepStatus) {
        setSettingsStatus(data);
      }
      if (data.scan_errors && data.scan_errors.length) {
        setSettingsStatus(data);
      }
    }

    async function refreshStatus() {
      const data = await apiJson("/health");
      lastStatus = data;
      document.getElementById("runtimeMeta").textContent = JSON.stringify(data, null, 2);
      document.getElementById("topbarModel").textContent = "Model: " + modelDisplayName(data);
      document.getElementById("modelId").value = data.model_id || "";
      document.getElementById("modelPath").value = data.model_path || "";
      document.getElementById("modelBaseDir").value = data.model_base_dir || "";
      document.getElementById("device").value = data.device || "";
      if (!document.getElementById("hfModelId").value) {
        document.getElementById("hfModelId").value = data.model_id || "";
      }
      await refreshModels(data.model_path || "", true);
    }

    async function saveSettings() {
      const payload = {
        model_id: document.getElementById("modelId").value || null,
        model_path: document.getElementById("modelPath").value || null,
        model_base_dir: document.getElementById("modelBaseDir").value || null,
        device: document.getElementById("device").value || null,
        persist_env: document.getElementById("persistEnv").checked
      };
      const data = await apiJson("/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setSettingsStatus(data);
      await refreshStatus();
    }

    async function executeLoad(payload) {
      const data = await apiJson("/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setSettingsStatus(data);
      appendMessage("assistant", "Model loaded: " + (data.model_source || data.model_id));
      await refreshStatus();
    }

    async function loadModel() {
      const payload = {
        model_id: document.getElementById("modelId").value || null,
        model_path: document.getElementById("modelPath").value || null,
        model_base_dir: document.getElementById("modelBaseDir").value || null,
        device: document.getElementById("device").value || null,
        force_reload: true,
        persist_env: document.getElementById("persistEnv").checked
      };

      const resolved = await apiJson("/model/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (resolved.local_expected && resolved.exists === false) {
        pendingLoadPayload = payload;
        openDownloadModal(resolved);
        switchSettingsTab("download");
        switchTab("settings");
        return;
      }

      await executeLoad(payload);
    }

    async function downloadFromHF() {
      const modelId = document.getElementById("hfModelId").value || document.getElementById("modelId").value;
      const payload = {
        model_id: modelId || null,
        base_dir: document.getElementById("modelBaseDir").value || null,
        subdir: document.getElementById("hfSubdir").value || null,
        revision: document.getElementById("hfRevision").value || null,
        allow_patterns: document.getElementById("hfAllowPatterns").value || null,
        ignore_patterns: document.getElementById("hfIgnorePatterns").value || null,
        set_as_active_model: document.getElementById("hfSetActive").checked,
        persist_env: document.getElementById("persistEnv").checked
      };
      const data = await apiJson("/hf/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setSettingsStatus(data);
      appendMessage("assistant", "Download queued: " + (data.downloaded_model_id || modelId));
      if (data.downloaded_model_id) {
        document.getElementById("modelId").value = data.downloaded_model_id;
      }
      if (data.downloaded_path) {
        document.getElementById("modelPath").value = data.downloaded_path;
      }
      if (data.job_id) {
        await pollDownloadJob(data.job_id);
      }
    }

    async function downloadFromHFModal() {
      const setActive = document.getElementById("modalHfSetActive").checked;
      const payload = {
        model_id: document.getElementById("modalHfModelId").value || document.getElementById("modelId").value || null,
        base_dir: document.getElementById("modelBaseDir").value || null,
        subdir: document.getElementById("modalHfSubdir").value || null,
        revision: document.getElementById("modalHfRevision").value || null,
        allow_patterns: document.getElementById("modalHfAllowPatterns").value || null,
        ignore_patterns: document.getElementById("modalHfIgnorePatterns").value || null,
        set_as_active_model: setActive,
        persist_env: document.getElementById("persistEnv").checked
      };
      const data = await apiJson("/hf/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setSettingsStatus(data);
      appendMessage("assistant", "Download queued: " + (data.downloaded_model_id || payload.model_id));
      if (data.job_id) {
        await pollDownloadJob(data.job_id, { closeModalOnComplete: true, triggerPendingLoad: setActive });
      }
    }

    async function sendPrompt() {
      if (isGenerating) {
        return;
      }
      const prompt = document.getElementById("prompt").value;
      if (!prompt || !prompt.trim()) {
        return;
      }
      appendMessage("user", prompt);
      document.getElementById("prompt").value = "";
      appendMessage("assistant", "Generating...");
      isGenerating = true;
      const payload = {
        prompt: prompt,
        max_new_tokens: Number(document.getElementById("maxNewTokens").value || 64),
        do_sample: asBool(document.getElementById("doSample").value)
      };
      try {
        const data = await apiJson("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const box = document.getElementById("messages");
        const last = box.lastElementChild;
        if (last && last.classList.contains("assistant") && last.textContent === "Generating...") {
          last.textContent = data.generated_text || data.text || "";
        } else {
          appendMessage("assistant", data.generated_text || data.text || "");
        }
        await refreshStatus();
      } catch (err) {
        const box = document.getElementById("messages");
        const last = box.lastElementChild;
        const errMsg = String(err);
        if (last && last.classList.contains("assistant") && last.textContent === "Generating...") {
          last.textContent = "Error: " + errMsg;
        } else {
          appendMessage("assistant", "Error: " + errMsg);
        }
      } finally {
        isGenerating = false;
      }
    }

    document.getElementById("downloadModal").addEventListener("click", (e) => {
      if (e.target && e.target.id === "downloadModal") {
        closeDownloadModal();
      }
    });
    document.getElementById("prompt").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeDownloadModal();
      }
    });

    initTheme();
    switchSettingsTab("config");
    setDownloadProgress({ job_id: "-", model_id: "-", target_dir: "-", status: "idle", progress: 0, current: 0, total: null, logs: ["No active downloads."] });
    const resumeJobId = localStorage.getItem("airllm_download_job_id");
    if (resumeJobId) {
      pollDownloadJob(resumeJobId).catch((err) => setSettingsStatus({ error: String(err) }));
    }
    refreshStatus().catch((err) => setSettingsStatus({ error: String(err) }));
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ui():
    state = _runtime_state()
    html = (
        UI_HTML
        .replace("__MODEL_ID_VALUE__", escape(state.get("model_id") or ""))
        .replace("__MODEL_PATH_VALUE__", escape(state.get("model_path") or ""))
        .replace("__MODEL_BASE_DIR_VALUE__", escape(state.get("model_base_dir") or ""))
        .replace("__DEVICE_VALUE__", escape(state.get("device") or ""))
    )
    return HTMLResponse(html)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON_PATH and FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH)
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/health")
def health():
    return {"status": "ok", **_runtime_state()}


@app.post("/model/resolve")
def resolve_model(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()
    target_model_id = _normalize_optional(payload.model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
    target_model_path = _normalize_optional(payload.model_path)
    if target_model_path is None:
        target_model_path = _runtime["model_path"]
    target_model_base_dir = _normalize_optional(payload.model_base_dir)
    if target_model_base_dir is None:
        target_model_base_dir = _runtime["model_base_dir"]

    target_source = _resolve_model_source(
        target_model_id,
        target_model_path,
        target_model_base_dir,
    )
    local_expected = bool(target_model_path or target_model_base_dir)
    exists = os.path.isdir(target_source) if local_expected else None
    return {
        "model_id": target_model_id,
        "model_path": target_model_path,
        "model_base_dir": target_model_base_dir,
        "model_source": target_source,
        "local_expected": local_expected,
        "exists": exists,
    }


@app.post("/settings")
def update_settings(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()

    target_model_id = _normalize_optional(payload.model_id) or _runtime["model_id"] or MODEL_ID_DEFAULT
    target_model_path = _normalize_optional(payload.model_path)
    if target_model_path is None:
        target_model_path = _runtime["model_path"]
    target_model_base_dir = _normalize_optional(payload.model_base_dir)
    if target_model_base_dir is None:
        target_model_base_dir = _runtime["model_base_dir"]
    target_device = _normalize_device(payload.device) or _runtime["device"] or DEVICE_DEFAULT
    target_model_source = _resolve_model_source(
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
    if payload.persist_env:
        _persist_runtime_to_env()

    return {
        "status": "settings_updated",
        **_runtime_state(),
        "persist_env": payload.persist_env,
        "env_file": str(ENV_FILE),
    }


@app.get("/models")
def list_models(base_dir: Optional[str] = None):
    target_base_dir = _normalize_optional(base_dir) or _runtime["model_base_dir"]
    if not target_base_dir:
        return {"base_dir": None, "models": [], "unsupported_models": [], "directories": [], "scan_errors": []}

    if not os.path.isdir(target_base_dir):
        return {
            "base_dir": target_base_dir,
            "models": [],
            "unsupported_models": [],
            "directories": [],
            "scan_errors": [f"Base directory does not exist or is inaccessible: {target_base_dir}"],
        }

    models, unsupported_models, scan_errors = _discover_models(target_base_dir)
    directories = _list_candidate_directories(target_base_dir)
    return {
        "base_dir": target_base_dir,
        "models": models,
        "unsupported_models": unsupported_models,
        "directories": directories,
        "scan_errors": scan_errors,
    }


@app.post("/hf/download")
def hf_download(request: HFDownloadRequest):
    model_id = _normalize_optional(request.model_id)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")

    target_base_dir = _normalize_optional(request.base_dir) or _runtime["model_base_dir"]
    if not target_base_dir:
        raise HTTPException(
            status_code=400,
            detail="No base directory set. Provide base_dir or configure AIRLLM_MODEL_BASE_DIR first.",
        )

    try:
        os.makedirs(target_base_dir, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to create/access base_dir: {exc}") from exc

    try:
        target_dir = _resolve_hf_download_dir(target_base_dir, model_id, request.subdir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "model_id": model_id,
        "base_dir": str(Path(target_base_dir).resolve()),
        "target_dir": str(target_dir),
        "revision": _normalize_optional(request.revision),
        "allow_patterns": _parse_patterns(request.allow_patterns),
        "ignore_patterns": _parse_patterns(request.ignore_patterns),
        "set_as_active_model": request.set_as_active_model,
        "persist_env": request.persist_env,
    }
    job_id = _new_download_job(model_id, payload["base_dir"], payload["target_dir"])
    thread = threading.Thread(target=_run_hf_download_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {
        "status": "started",
        "job_id": job_id,
        "downloaded_model_id": model_id,
        "downloaded_path": str(target_dir),
        "base_dir": payload["base_dir"],
        "set_as_active_model": request.set_as_active_model,
        "persist_env": request.persist_env,
        "env_file": str(ENV_FILE),
    }


@app.get("/hf/download/{job_id}")
def hf_download_status(job_id: str):
    job = _job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No download job found for id {job_id}")
    return job


@app.post("/load")
def preload_model(request: Optional[LoadRequest] = Body(default=None)):
    payload = request or LoadRequest()
    try:
        load_model(
            model_id=payload.model_id,
            model_path=payload.model_path,
            model_base_dir=payload.model_base_dir,
            device=payload.device,
            force_reload=payload.force_reload,
        )
        if payload.persist_env:
            _persist_runtime_to_env()
        return {
            "status": "loaded",
            **_runtime_state(),
            "persist_env": payload.persist_env,
            "env_file": str(ENV_FILE),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        model = load_model()

        input_tokens = model.tokenizer(
            [request.prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=False,
        )

        input_ids = input_tokens["input_ids"].to(_runtime["device"])
        generate_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if request.do_sample:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                }
            )

        output = model.generate(input_ids, **generate_kwargs)
        full_text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)

        if full_text.startswith(request.prompt):
            generated = full_text[len(request.prompt) :].lstrip()
        else:
            generated = full_text

        return GenerateResponse(
            model_id=_runtime["model_id"],
            model_source=_runtime["model_source"],
            device=_runtime["device"],
            text=full_text,
            generated_text=generated,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
