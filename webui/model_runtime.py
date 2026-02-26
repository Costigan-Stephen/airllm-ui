import io
import gc
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import torch
from dotenv import dotenv_values, set_key

# Always prioritize the local bundled air_llm package so runtime behavior
# matches repository code changes and does not depend on stale site-packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_AIR_LLM_ROOT = _PROJECT_ROOT / "air_llm"
if _LOCAL_AIR_LLM_ROOT.exists():
    local_pkg_path = str(_LOCAL_AIR_LLM_ROOT)
    if local_pkg_path not in sys.path:
        sys.path.insert(0, local_pkg_path)

from airllm import AutoModel

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None
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
    "runtime_backend": None,
    "airllm_optimized": None,
    "runtime_note": None,
}

_load_jobs = {}
_load_jobs_lock = threading.Lock()

_SHARD_RE = re.compile(r"loading shard\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LAYERS_RE = re.compile(r"running layers[^\r\n]*?(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_COUNT_RE = re.compile(r"(?<![\d:])(\d{1,5})\s*/\s*(\d{1,5})(?![\d:])")
_PERCENT_RE = re.compile(r"(\d{1,3})%")


class _TransformersRuntimeModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = getattr(model, "generation_config", None)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)



class _LlamaCppRuntimeModel:
    def __init__(self, llm, model_path: str, uses_gpu_layers: bool):
        self.llm = llm
        self.model_path = model_path
        self.uses_gpu_layers = uses_gpu_layers

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        kwargs = {
            "prompt": prompt,
            "max_tokens": int(max_new_tokens),
            "echo": False,
            "stop": [],
        }
        if do_sample:
            kwargs.update(
                {
                    "temperature": float(max(0.0, temperature)),
                    "top_p": float(max(0.0, min(1.0, top_p))),
                    "top_k": int(max(1, top_k)),
                }
            )
        else:
            kwargs.update(
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 40,
                }
            )

        result = self.llm.create_completion(**kwargs)
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return ""
        return str(choices[0].get("text") or "")

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
            "runtime_backend": _runtime.get("runtime_backend"),
            "airllm_optimized": _runtime.get("airllm_optimized"),
            "runtime_note": _runtime.get("runtime_note"),
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


def _runtime_env_snapshot():
    with _model_lock:
        return {
            "model_id": _runtime["model_id"] or "",
            "model_path": _runtime["model_path"] or "",
            "model_base_dir": _runtime["model_base_dir"] or "",
            "device": _runtime["device"] or "",
        }


def _saved_env_snapshot():
    if not ENV_FILE.exists():
        return {
            "model_id": "",
            "model_path": "",
            "model_base_dir": "",
            "device": "",
        }

    env_values = dotenv_values(str(ENV_FILE))
    return {
        "model_id": (env_values.get("AIRLLM_MODEL_ID") or "").strip(),
        "model_path": (env_values.get("AIRLLM_MODEL_PATH") or "").strip(),
        "model_base_dir": (env_values.get("AIRLLM_MODEL_BASE_DIR") or "").strip(),
        "device": (env_values.get("AIRLLM_DEVICE") or "").strip(),
    }


def runtime_differs_from_env() -> bool:
    return _runtime_env_snapshot() != _saved_env_snapshot()


def persist_runtime_to_env():
    snapshot = _runtime_env_snapshot()

    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), "AIRLLM_MODEL_ID", snapshot["model_id"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_PATH", snapshot["model_path"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_MODEL_BASE_DIR", snapshot["model_base_dir"], quote_mode="never")
    set_key(str(ENV_FILE), "AIRLLM_DEVICE", snapshot["device"], quote_mode="never")


def persist_runtime_to_env_if_changed() -> bool:
    if not runtime_differs_from_env():
        return False
    persist_runtime_to_env()
    return True


def _dispose_model_instance(model_obj):
    if model_obj is None:
        return

    try:
        if isinstance(model_obj, _LlamaCppRuntimeModel):
            llm_obj = getattr(model_obj, "llm", None)
            close_fn = getattr(llm_obj, "close", None)
            if callable(close_fn):
                close_fn()
    except Exception:
        pass

    try:
        if isinstance(model_obj, _TransformersRuntimeModel):
            hf_model = getattr(model_obj, "model", None)
            if hf_model is not None:
                hf_model.cpu()
    except Exception:
        pass


def _cleanup_model(model_obj=None):
    _dispose_model_instance(model_obj)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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


def _normalize_runtime_backend(value: Optional[str]) -> Optional[str]:
    candidate = normalize_optional(value)
    if not candidate or candidate == "auto":
        return None
    candidate = candidate.lower()
    if candidate == "llama.cpp":
        candidate = "llama_cpp"
    if candidate not in ("airllm", "transformers", "llama_cpp"):
        raise ValueError(f"Unsupported runtime backend: {candidate}")
    return candidate


def _find_local_gguf_path(source_for_load: str) -> Optional[str]:
    source_path = Path(source_for_load)
    if source_path.is_file() and source_path.suffix.lower() == ".gguf":
        return str(source_path)

    if not source_path.is_dir():
        return None

    gguf_files = list(source_path.glob("*.gguf"))
    if not gguf_files:
        gguf_files = list(source_path.rglob("*.gguf"))
    if not gguf_files:
        return None

    # Prefer the largest file when multiple quant variants exist.
    gguf_files.sort(key=lambda path_value: path_value.stat().st_size if path_value.exists() else 0, reverse=True)
    return str(gguf_files[0])


def _llama_cpp_backend_available() -> bool:
    return Llama is not None


def _load_llama_cpp_model(source_for_load: str, target_device: str):
    gguf_path = _find_local_gguf_path(source_for_load)
    if not gguf_path:
        raise RuntimeError(
            f"No GGUF file found in {source_for_load}. llama.cpp fallback requires a local .gguf model file."
        )

    if Llama is None:
        raise RuntimeError(
            "llama.cpp backend detected, but python bindings are unavailable. "
            "Install llama-cpp-python in this environment to enable fallback generation."
        )

    n_ctx = int(os.getenv("AIRLLM_LLAMA_CPP_N_CTX", "4096"))
    use_gpu = target_device.startswith("cuda") and torch.cuda.is_available()
    n_gpu_layers = -1 if use_gpu else 0

    try:
        llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        return _LlamaCppRuntimeModel(llm, gguf_path, uses_gpu_layers=(n_gpu_layers != 0))
    except Exception:
        if n_gpu_layers == 0:
            raise
        # Retry CPU path if GPU init fails.
        llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=0, verbose=False)
        return _LlamaCppRuntimeModel(llm, gguf_path, uses_gpu_layers=False)

def _load_airllm_model(source_for_load: str, target_device: str):
    kwargs = {"device": target_device}
    if HF_TOKEN:
        kwargs["hf_token"] = HF_TOKEN
    return AutoModel.from_pretrained(source_for_load, **kwargs)


def _load_transformers_model(source_for_load: str, target_device: str):
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "Transformers runtime fallback is unavailable because AutoModelForCausalLM/AutoTokenizer could not be imported."
        )

    token_kwargs = {}
    if HF_TOKEN:
        token_kwargs["token"] = HF_TOKEN

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source_for_load,
            trust_remote_code=True,
            **token_kwargs,
        )
    except TypeError:
        if not HF_TOKEN:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            source_for_load,
            trust_remote_code=True,
            use_auth_token=HF_TOKEN,
        )

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN
    if target_device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = "auto"

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(source_for_load, **model_kwargs)
    except TypeError:
        if "token" in model_kwargs:
            model_kwargs.pop("token", None)
            model_kwargs["use_auth_token"] = HF_TOKEN
        hf_model = AutoModelForCausalLM.from_pretrained(source_for_load, **model_kwargs)
    except Exception as exc:
        message = str(exc)
        if "does not recognize this architecture" in message or "model_type" in message:
            raise RuntimeError(
                "Transformers cannot load this model architecture with the currently installed version. "
                "Update Transformers to a version that supports this model_type or use a model with known support. "
                f"Original error: {message}"
            ) from exc
        raise

    if target_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {target_device}, but CUDA is not available.")

    hf_model.to(target_device)
    hf_model.eval()
    return _TransformersRuntimeModel(hf_model, tokenizer)


def load_model(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_base_dir: Optional[str] = None,
    device: Optional[str] = None,
    force_reload: bool = False,
    runtime_backend: Optional[str] = None,
):
    global _model

    preferred_backend = _normalize_runtime_backend(runtime_backend)

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
            and (preferred_backend is None or _runtime.get("runtime_backend") == preferred_backend)
        )
        if same_config:
            if not force_reload:
                return _model
            # Avoid repeatedly respawning llama.cpp backends for identical config.
            if _runtime.get("runtime_backend") == "llama_cpp":
                return _model

        if _model is not None:
            previous_model = _model
            _model = None
            _cleanup_model(previous_model)

        local_expected = bool(
            target_model_path
            or target_model_base_dir
            or _looks_like_local_path(target_model_source)
        )

        source_for_load = target_model_source
        local_details = None
        if local_expected:
            local_source = os.path.expanduser(target_model_source)
            if not os.path.exists(local_source):
                raise FileNotFoundError(f"Local model path does not exist: {target_model_source}")
            try:
                local_details = validate_local_model_source(local_source)
            except ValueError as validation_error:
                validation_message = str(validation_error)
                if "not supported by installed transformers" in validation_message:
                    # Do not hard-block local checkpoints solely on model_type registry checks.
                    # We still attempt standard Transformers loading with trust_remote_code,
                    # then fallback to llama.cpp if GGUF is present.
                    local_details = {
                        "runtime_mode": "transformers",
                        "airllm_optimized": False,
                        "optimization_note": (
                            "Model architecture is newer than installed Transformers registry. "
                            "Attempting standard Transformers runtime and llama.cpp fallback if available."
                        ),
                        "warnings": [validation_message],
                    }
                elif "GGUF" in validation_message:
                    local_details = {
                        "runtime_mode": "llama_cpp",
                        "airllm_optimized": False,
                        "optimization_note": "GGUF model detected. Using llama.cpp backend.",
                        "warnings": [validation_message],
                    }
                else:
                    raise
            source_for_load = local_source

        if preferred_backend is None:
            if isinstance(local_details, dict):
                preferred_backend = local_details.get("runtime_mode") or "airllm"
            else:
                preferred_backend = "airllm"

        gguf_candidate = _find_local_gguf_path(source_for_load) if local_expected else None
        llama_cpp_available = _llama_cpp_backend_available()
        allow_llama_cpp = bool(local_expected and gguf_candidate and llama_cpp_available)

        llama_cpp_unavailable_reason = None
        if local_expected and not gguf_candidate:
            llama_cpp_unavailable_reason = (
                "No local .gguf found for llama.cpp fallback in the selected model directory."
            )
        elif local_expected and gguf_candidate and not llama_cpp_available:
            llama_cpp_unavailable_reason = (
                "llama.cpp fallback is available only when llama-cpp-python is installed in this environment."
            )

        if preferred_backend == "airllm":
            backends_to_try = ["airllm", "transformers"]
            if allow_llama_cpp:
                backends_to_try.append("llama_cpp")
        elif preferred_backend == "llama_cpp":
            backends_to_try = ["llama_cpp"] if allow_llama_cpp else ["transformers"]
        else:
            backends_to_try = ["transformers"]
            if allow_llama_cpp:
                backends_to_try.append("llama_cpp")

        last_error = None
        load_errors = []
        loaded_backend = None
        loaded_runtime_note = None
        loaded_model = None
        for backend_name in backends_to_try:
            try:
                if backend_name == "airllm":
                    if local_expected:
                        _ensure_airllm_index_files(source_for_load)
                    candidate_model = _load_airllm_model(source_for_load, target_device)
                    candidate_note = "AirLLM optimization enabled."
                    if isinstance(local_details, dict) and local_details.get("warnings"):
                        candidate_note = candidate_note + " " + " ".join(local_details.get("warnings") or [])
                elif backend_name == "llama_cpp":
                    candidate_model = _load_llama_cpp_model(source_for_load, target_device)
                    gguf_name = Path(candidate_model.model_path).name
                    if candidate_model.uses_gpu_layers:
                        candidate_note = f"Using llama.cpp backend with GPU offload ({gguf_name})."
                    else:
                        candidate_note = f"Using llama.cpp backend on CPU ({gguf_name})."
                else:
                    candidate_model = _load_transformers_model(source_for_load, target_device)
                    candidate_note = "Using standard Transformers runtime (AirLLM optimization unavailable for this model)."
                    if isinstance(local_details, dict) and local_details.get("optimization_note"):
                        candidate_note = str(local_details.get("optimization_note"))
                    if llama_cpp_unavailable_reason and "llama.cpp" not in candidate_note.lower():
                        candidate_note = candidate_note + " " + llama_cpp_unavailable_reason
                    if load_errors:
                        candidate_note = candidate_note + " Fallback reason: " + load_errors[-1]

                loaded_model = candidate_model
                loaded_backend = backend_name
                loaded_runtime_note = candidate_note
                break
            except Exception as exc:
                last_error = exc
                load_errors.append(f"{backend_name} load failed: {exc}")

        if loaded_model is None:
            if llama_cpp_unavailable_reason and not allow_llama_cpp:
                load_errors.append(f"llama_cpp skipped: {llama_cpp_unavailable_reason}")
            if load_errors:
                raise RuntimeError("Model load failed. " + " | ".join(load_errors)) from last_error
            raise RuntimeError("Model load failed for unknown reason.")

        _model = loaded_model
        _runtime.update(
            {
                "model_id": target_model_id,
                "model_path": target_model_path,
                "model_base_dir": target_model_base_dir,
                "device": target_device,
                "model_source": source_for_load,
                "runtime_backend": loaded_backend,
                "airllm_optimized": loaded_backend == "airllm",
                "runtime_note": loaded_runtime_note,
            }
        )

    return _model


def apply_download_result(model_id: str, base_dir: str, target_dir: str, set_as_active_model: bool = True):
    global _model

    with _model_lock:
        _runtime["model_base_dir"] = base_dir
        if set_as_active_model:
            if _model is not None:
                previous_model = _model
                _model = None
                _cleanup_model(previous_model)
            _runtime.update(
                {
                    "model_id": model_id,
                    "model_path": target_dir,
                    "model_source": target_dir,
                    "runtime_backend": None,
                    "airllm_optimized": None,
                    "runtime_note": None,
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
            "updated_at": time.time(),
            "finished_at": None,
            "model_id": payload.get("model_id"),
            "model_path": payload.get("model_path"),
            "model_base_dir": payload.get("model_base_dir"),
            "device": payload.get("device"),
            "persist_env": bool(payload.get("persist_env", False)),
            "cancel_requested": False,
            "result": None,
        }
    return job_id


def _load_job_update(job_id: str, **fields):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()


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
        job["updated_at"] = time.time()


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
            "updated_at": job.get("updated_at"),
        }



def _is_load_cancel_requested(job_id: str) -> bool:
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return False
        return bool(job.get("cancel_requested", False))


def request_stop_load_job(job_id: Optional[str] = None):
    with _load_jobs_lock:
        target_id = job_id
        if not target_id:
            active = [job for job in _load_jobs.values() if job.get("status") in ("queued", "loading")]
            if not active:
                return {"status": "idle", "stopped": False, "job_id": None}
            active.sort(key=lambda item: item.get("started_at", 0), reverse=True)
            target_id = active[0].get("job_id")

        job = _load_jobs.get(target_id)
        if not job:
            return {"status": "not_found", "stopped": False, "job_id": target_id}


        if job.get("status") == "queued":
            job["cancel_requested"] = True
            job["status"] = "canceled"
            job["stage"] = "canceled"
            job["message"] = "Model load canceled before start."
            job["finished_at"] = time.time()
            logs = job.get("logs", [])
            logs.append("Model load canceled before start.")
            if len(logs) > 300:
                logs = logs[-300:]
            job["logs"] = logs
            return {"status": "canceled", "stopped": True, "job_id": target_id}

        if job.get("status") not in ("queued", "loading"):
            return {
                "status": "not_active",
                "stopped": False,
                "job_id": target_id,
                "job_status": job.get("status"),
            }

        job["cancel_requested"] = True
        job["message"] = "Cancellation requested..."
        logs = job.get("logs", [])
        logs.append("Cancellation requested by user.")
        if len(logs) > 300:
            logs = logs[-300:]
        job["logs"] = logs

    return {"status": "cancel_requested", "stopped": True, "job_id": target_id}

def _update_load_progress(job_id: str, *, stage: Optional[str] = None, current: Optional[int] = None,
                          total: Optional[int] = None, progress: Optional[int] = None, message: Optional[str] = None):
    with _load_jobs_lock:
        job = _load_jobs.get(job_id)
        if not job:
            return

        old_stage = job.get("stage")
        old_progress = int(job.get("progress") or 0)
        job_status = job.get("status")
        next_stage = stage if stage is not None else old_stage

        if stage is not None:
            job["stage"] = stage
        if current is not None:
            job["current"] = int(current)
        if total is not None:
            job["total"] = int(total)

        if progress is not None:
            next_progress = max(0, min(100, int(progress)))
            if job_status in ("queued", "loading") and next_stage not in ("completed", "failed", "canceled"):
                # Keep in-flight jobs below 100%. 100% is reserved for true completion.
                next_progress = min(next_progress, 99)
            if next_stage and old_stage and next_stage == old_stage and next_progress < old_progress:
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

        job["updated_at"] = time.time()


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
        stage = "loading_shards"
        message = f"Loading shard {current}/{total}"
        if total and current >= total:
            stage = "finalizing"
            message = f"Loaded shards {current}/{total}. Finalizing model setup..."
            pct = 99
        _update_load_progress(
            job_id,
            stage=stage,
            current=current,
            total=total,
            progress=pct,
            message=message,
        )
        return

    layers_match = _LAYERS_RE.search(clean)
    if layers_match:
        current = int(layers_match.group(1))
        total = int(layers_match.group(2))
        pct_match = _PERCENT_RE.search(clean)
        pct = int(pct_match.group(1)) if pct_match else (int((current / total) * 100) if total else 0)
        stage = "loading_layers"
        message = f"Loading layers {current}/{total}"
        if total and current >= total:
            stage = "finalizing"
            message = f"Loaded layers {current}/{total}. Finalizing model setup..."
            pct = 99
        _update_load_progress(
            job_id,
            stage=stage,
            current=current,
            total=total,
            progress=pct,
            message=message,
        )
        return

    count_match = _COUNT_RE.search(clean)
    pct_match = _PERCENT_RE.search(clean)
    if count_match and ("layer" in clean.lower() or "shard" in clean.lower()):
        current = int(count_match.group(1))
        total = int(count_match.group(2))
        pct = int(pct_match.group(1)) if pct_match else (int((current / total) * 100) if total else 0)
        is_shard = "shard" in clean.lower()
        stage = "loading_shards" if is_shard else "loading_layers"
        label = "Loading shards" if is_shard else "Loading layers"
        message = f"{label} {current}/{total}"
        if total and current >= total:
            stage = "finalizing"
            message = f"Loaded {'shards' if is_shard else 'layers'} {current}/{total}. Finalizing model setup..."
            pct = 99
        _update_load_progress(
            job_id,
            stage=stage,
            current=current,
            total=total,
            progress=pct,
            message=message,
        )
        return

    if pct_match:
        pct = int(pct_match.group(1))
        stage = "loading"
        message = clean
        if pct >= 100:
            stage = "finalizing"
            pct = 99
            message = "Weights loaded. Finalizing model setup..."
        _update_load_progress(
            job_id,
            stage=stage,
            progress=pct,
            message=message,
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

        if not _is_load_cancel_requested(self._job_id):
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
    global _model

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
        if _is_load_cancel_requested(job_id):
            _load_job_update(
                job_id,
                status="canceled",
                stage="canceled",
                message="Model load canceled by user.",
                error=None,
                finished_at=time.time(),
            )
            _load_job_log(job_id, "Model load canceled before start.")
            return

        with redirect_stdout(out_stream), redirect_stderr(err_stream):
            load_model(
                model_id=payload.get("model_id"),
                model_path=payload.get("model_path"),
                model_base_dir=payload.get("model_base_dir"),
                device=payload.get("device"),
                force_reload=bool(payload.get("force_reload", False)),
            )

        if _is_load_cancel_requested(job_id):
            with _model_lock:
                if _model is not None:
                    previous_model = _model
                    _model = None
                    _cleanup_model(previous_model)
                _runtime["runtime_backend"] = None
                _runtime["airllm_optimized"] = None
                _runtime["runtime_note"] = None
            _load_job_update(
                job_id,
                status="canceled",
                stage="canceled",
                message="Model load canceled by user.",
                error=None,
                finished_at=time.time(),
            )
            _load_job_log(job_id, "Model load canceled by user. Loaded model was discarded.")
            return

        persisted_env = False
        if bool(payload.get("persist_env", False)):
            persist_runtime_to_env()
            persisted_env = True
            _load_job_log(job_id, f"Persisted settings to {ENV_FILE}")
        else:
            persisted_env = persist_runtime_to_env_if_changed()
            if persisted_env:
                _load_job_log(job_id, f"Auto-persisted loaded settings to {ENV_FILE} (configuration changed).")

        snapshot = runtime_state()
        snapshot["persist_env_requested"] = bool(payload.get("persist_env", False))
        snapshot["persisted_env"] = bool(persisted_env)
        snapshot["env_file"] = str(ENV_FILE)
        _load_job_update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Model loaded",
            error=None,
            finished_at=time.time(),
            persisted_env=persisted_env,
            result=snapshot,
        )
        _load_job_log(job_id, "Model load completed.")
    except Exception as exc:
        exc_text = str(exc)
        if "load canceled by user" in exc_text.lower():
            _load_job_update(
                job_id,
                status="canceled",
                stage="canceled",
                message="Model load canceled by user.",
                error=None,
                finished_at=time.time(),
            )
            _load_job_log(job_id, "Model load canceled by user.")
            return

        _load_job_update(
            job_id,
            status="failed",
            stage="failed",
            message=f"Model load failed: {exc}",
            error=exc_text,
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

























