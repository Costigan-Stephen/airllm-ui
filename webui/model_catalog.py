import json
import os
from pathlib import Path


def inspect_local_model_dir(root_path: Path, files, exhaustive: bool = False):
    file_set = set(files)
    has_gguf = any(name.lower().endswith(".gguf") for name in files)
    has_config = "config.json" in file_set
    has_airllm_index = (
        "pytorch_model.bin.index.json" in file_set
        or "model.safetensors.index.json" in file_set
    )
    has_weights = (
        "pytorch_model.bin" in file_set
        or "pytorch_model.bin.index.json" in file_set
        or "model.safetensors.index.json" in file_set
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
    if has_config and model_type and has_weights and not has_airllm_index:
        return (
            False,
            "AirLLM requires a sharded checkpoint index file "
            "(model.safetensors.index.json or pytorch_model.bin.index.json).",
        )
    if has_config and model_type and has_weights:
        return True, f"model_type={model_type}"
    if exhaustive:
        return False, "No compatible Transformers config.json + weights found."
    return False, None


def validate_local_model_source(model_source: str):
    source_path = Path(model_source)
    if source_path.is_file() and source_path.suffix.lower() == ".gguf":
        raise ValueError(
            f"Selected model is GGUF ({model_source}). AirLLM expects a Hugging Face Transformers checkpoint directory."
        )
    if not source_path.is_dir():
        return

    files = [entry.name for entry in source_path.iterdir() if entry.is_file()]
    compatible, reason = inspect_local_model_dir(source_path, files, exhaustive=True)
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
    if reason and "sharded checkpoint index file" in reason:
        raise ValueError(
            f"Selected local model path is missing AirLLM-required index files: {model_source} ({reason})"
        )
    if reason:
        raise ValueError(
            f"Selected local model path is not compatible with AirLLM: {model_source} ({reason})"
        )


def discover_models(base_dir: str, max_depth: int = 4, limit: int = 500):
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

        compatible, reason = inspect_local_model_dir(root_path, files)
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


def list_candidate_directories(base_dir: str, max_depth: int = 2, limit: int = 300):
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
