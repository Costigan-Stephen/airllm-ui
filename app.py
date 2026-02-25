import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from webui.config import ENV_FILE, MAX_INPUT_TOKENS, PORT, current_favicon_path
from webui.downloads import (
    active_download,
    download_job_snapshot,
    parse_patterns,
    resolve_hf_download_dir,
    start_hf_download_job,
)
from webui.model_catalog import discover_models, list_candidate_directories
from webui.model_runtime import (
    apply_settings,
    active_load_job,
    get_runtime_field,
    load_model,
    load_job_snapshot,
    looks_like_local_path,
    normalize_optional,
    persist_runtime_to_env,
    resolve_requested_source,
    start_load_job,
    runtime_state,
)
from webui.schemas import (
    GenerateRequest,
    GenerateResponse,
    HFDownloadRequest,
    LoadRequest,
    SettingsRequest,
)
from webui.ui import render_ui_html

app = FastAPI(title="AirLLM API", version="1.0.0")
_autoload_lock = threading.Lock()
_autoload_attempted_source = None
_autoload_last_error = None


def _coerce_local_path_input(model_id: Optional[str], model_path: Optional[str]):
    normalized_model_id = normalize_optional(model_id)
    normalized_model_path = normalize_optional(model_path)
    if normalized_model_path:
        return normalized_model_id, normalized_model_path
    if normalized_model_id and looks_like_local_path(normalized_model_id):
        return None, normalized_model_id
    return normalized_model_id, normalized_model_path


def _maybe_autoload_configured_local_model():
    global _autoload_attempted_source, _autoload_last_error

    state = runtime_state()
    if state.get("model_loaded"):
        return {"status": "already_loaded"}

    resolved = resolve_requested_source()
    model_source = resolved.get("model_source")
    local_expected = bool(resolved.get("model_path") or resolved.get("model_base_dir"))
    if not local_expected:
        return {"status": "skipped_remote"}
    if not model_source or not os.path.isdir(model_source):
        return {"status": "skipped_missing_local", "model_source": model_source}

    with _autoload_lock:
        if _autoload_attempted_source == model_source and _autoload_last_error is not None:
            return {
                "status": "skipped_previous_failure",
                "model_source": model_source,
                "error": _autoload_last_error,
            }
        _autoload_attempted_source = model_source
        _autoload_last_error = None

    try:
        load_model(force_reload=False)
        with _autoload_lock:
            _autoload_last_error = None
        return {"status": "loaded", "model_source": model_source}
    except Exception as exc:
        err = str(exc)
        with _autoload_lock:
            _autoload_last_error = err
        print(f"[autoloader] Failed to load configured model: {err}")
        traceback.print_exc()
        return {"status": "failed", "model_source": model_source, "error": err}


@app.on_event("startup")
def startup_autoload():
    _maybe_autoload_configured_local_model()


@app.get("/", response_class=HTMLResponse)
def ui():
    _maybe_autoload_configured_local_model()
    html = render_ui_html(runtime_state())
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = current_favicon_path()
    if favicon_path and favicon_path.exists():
        return FileResponse(favicon_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/favicon", include_in_schema=False)
def favicon_any():
    favicon_path = current_favicon_path()
    if favicon_path and favicon_path.exists():
        return FileResponse(favicon_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/health")
def health():
    airllm_module = None
    try:
        import airllm as _airllm_pkg
        airllm_module = getattr(_airllm_pkg, "__file__", None)
    except Exception:
        airllm_module = None
    return {
        "status": "ok",
        **runtime_state(),
        "app_file": __file__,
        "airllm_module": airllm_module,
    }


@app.post("/model/resolve")
def resolve_model(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    resolved = resolve_requested_source(
        model_id=model_id,
        model_path=model_path,
        model_base_dir=payload.model_base_dir,
    )
    local_expected = bool(resolved["model_path"] or resolved["model_base_dir"])
    exists = os.path.isdir(resolved["model_source"]) if local_expected else None
    return {
        **resolved,
        "local_expected": local_expected,
        "exists": exists,
    }


@app.post("/settings")
def update_settings(request: Optional[SettingsRequest] = Body(default=None)):
    payload = request or SettingsRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    apply_settings(
        model_id=model_id,
        model_path=model_path,
        model_base_dir=payload.model_base_dir,
        device=payload.device,
    )
    if payload.persist_env:
        persist_runtime_to_env()

    return {
        "status": "settings_updated",
        **runtime_state(),
        "persist_env": payload.persist_env,
        "env_file": str(ENV_FILE),
    }


@app.get("/models")
def list_models(base_dir: Optional[str] = None):
    target_base_dir = normalize_optional(base_dir) or get_runtime_field("model_base_dir")
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

    models, unsupported_models, scan_errors = discover_models(target_base_dir)
    directories = list_candidate_directories(target_base_dir)
    return {
        "base_dir": target_base_dir,
        "models": models,
        "unsupported_models": unsupported_models,
        "directories": directories,
        "scan_errors": scan_errors,
    }


@app.post("/hf/download")
def hf_download(request: HFDownloadRequest):
    model_id = normalize_optional(request.model_id)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")

    target_base_dir = normalize_optional(request.base_dir) or get_runtime_field("model_base_dir")
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
        target_dir = resolve_hf_download_dir(target_base_dir, model_id, request.subdir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "model_id": model_id,
        "base_dir": str(Path(target_base_dir).resolve()),
        "target_dir": str(target_dir),
        "revision": normalize_optional(request.revision),
        "allow_patterns": parse_patterns(request.allow_patterns),
        "ignore_patterns": parse_patterns(request.ignore_patterns),
        "set_as_active_model": request.set_as_active_model,
        "persist_env": request.persist_env,
    }
    job_id = start_hf_download_job(payload)
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


@app.get("/hf/download/active")
def hf_download_active():
    return active_download()


@app.get("/hf/download/{job_id}")
def hf_download_status(job_id: str):
    job = download_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No download job found for id {job_id}")
    return job


@app.post("/load/start")
def preload_model_start(request: Optional[LoadRequest] = Body(default=None)):
    payload = request or LoadRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    job_id = start_load_job(
        {
            "model_id": model_id,
            "model_path": model_path,
            "model_base_dir": payload.model_base_dir,
            "device": payload.device,
            "force_reload": payload.force_reload,
            "persist_env": payload.persist_env,
        }
    )
    return {
        "status": "started",
        "job_id": job_id,
    }


@app.get("/load/active")
def load_active():
    return active_load_job()


@app.get("/load/{job_id}")
def load_status(job_id: str):
    job = load_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No load job found for id {job_id}")
    return job


@app.post("/load")
def preload_model(request: Optional[LoadRequest] = Body(default=None)):
    payload = request or LoadRequest()
    model_id, model_path = _coerce_local_path_input(payload.model_id, payload.model_path)
    try:
        load_model(
            model_id=model_id,
            model_path=model_path,
            model_base_dir=payload.model_base_dir,
            device=payload.device,
            force_reload=payload.force_reload,
        )
        if payload.persist_env:
            persist_runtime_to_env()
        return {
            "status": "loaded",
            **runtime_state(),
            "persist_env": payload.persist_env,
            "env_file": str(ENV_FILE),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        model = load_model()
        runtime = runtime_state()
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False

        input_tokens = model.tokenizer(
            [request.prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=False,
        )

        input_ids = input_tokens["input_ids"].to(runtime["device"])
        generate_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "use_cache": False,
        }
        if request.do_sample:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                }
            )        try:
            output = model.generate(input_ids, **generate_kwargs)
        except TypeError as exc:
            if "cannot unpack non-iterable NoneType object" not in str(exc):
                raise
            traceback.print_exc()
            fallback_kwargs = {
                "max_new_tokens": request.max_new_tokens,
            }
            if request.do_sample:
                fallback_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                    }
                )
            output = model.generate(input_ids, **fallback_kwargs)

        if hasattr(output, "sequences"):
            sequence = output.sequences[0]
        elif isinstance(output, tuple):
            sequence = output[0]
        else:
            sequence = output
        if getattr(sequence, "ndim", 0) > 1:
            sequence = sequence[0]

        full_text = model.tokenizer.decode(sequence, skip_special_tokens=True)
        generated = full_text[len(request.prompt) :].lstrip() if full_text.startswith(request.prompt) else full_text

        runtime = runtime_state()
        return GenerateResponse(
            model_id=runtime["model_id"],
            model_source=runtime["model_source"],
            device=runtime["device"],
            text=full_text,
            generated_text=generated,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(exc).__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)







