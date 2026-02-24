import os
import threading
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

