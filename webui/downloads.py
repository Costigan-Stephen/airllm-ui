import os
import threading
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download

from .config import ENV_FILE, HF_TOKEN
from .model_runtime import apply_download_result, normalize_optional, persist_runtime_to_env

_download_jobs = {}
_download_lock = threading.Lock()


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
            "current_file": None,
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
        if len(logs) > 300:
            logs = logs[-300:]
        job["logs"] = logs


def parse_patterns(patterns: Optional[str]):
    patterns = normalize_optional(patterns)
    if not patterns:
        return None
    values = []
    for chunk in patterns.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values or None


def resolve_hf_download_dir(base_dir: str, model_id: str, subdir: Optional[str]):
    model_id = model_id.strip()
    base = Path(base_dir)
    if subdir:
        rel = Path(subdir.strip())
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("subdir must be a relative path under the base directory.")
        return (base / rel).resolve()
    return (base / Path(*model_id.split("/"))).resolve()


def _run_hf_download_job(job_id: str, payload: dict):
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
    _job_update(job_id, progress=1, current=0, total=None, current_file="Resolving repository file list")

    def _matches_any(path_value: str, patterns):
        if not patterns:
            return False
        return any(fnmatch(path_value, pattern) for pattern in patterns)

    try:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        api = HfApi(token=HF_TOKEN)
        repo_files = list(api.list_repo_files(repo_id=model_id, revision=revision))
        repo_files = [path_value for path_value in repo_files if not path_value.endswith("/")]

        if allow_patterns:
            repo_files = [path_value for path_value in repo_files if _matches_any(path_value, allow_patterns)]
        if ignore_patterns:
            repo_files = [path_value for path_value in repo_files if not _matches_any(path_value, ignore_patterns)]

        total_files = len(repo_files)
        if total_files == 0:
            raise RuntimeError("No files matched the selected patterns for download.")

        _job_update(job_id, total=total_files, current=0, progress=1)
        _job_log(job_id, f"Resolved {total_files} files to download.")

        done = 0
        last_log = 0.0
        for repo_file in repo_files:
            progress = max(1, int((done / total_files) * 100))
            _job_update(
                job_id,
                current=done,
                total=total_files,
                progress=progress,
                current_file=repo_file,
            )
            hf_hub_download(
                repo_id=model_id,
                filename=repo_file,
                revision=revision,
                local_dir=target_dir,
                token=HF_TOKEN,
            )
            done += 1
            progress = int((done / total_files) * 100)
            _job_update(job_id, current=done, total=total_files, progress=progress, current_file=repo_file)
            now = time.time()
            if now - last_log >= 0.5 or done == total_files:
                _job_log(job_id, f"Download progress: {done}/{total_files} files")
                last_log = now
    except Exception as exc:
        _job_log(job_id, f"Hugging Face download failed: {exc}")
        _job_update(job_id, status="failed", error=str(exc), finished_at=time.time())
        return

    _job_log(job_id, "Download finished successfully.")
    apply_download_result(
        model_id=model_id,
        base_dir=base_dir,
        target_dir=target_dir,
        set_as_active_model=set_as_active_model,
    )

    if persist_env:
        persist_runtime_to_env()
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


def start_hf_download_job(payload: dict):
    job_id = _new_download_job(
        model_id=payload["model_id"],
        base_dir=payload["base_dir"],
        target_dir=payload["target_dir"],
    )
    thread = threading.Thread(target=_run_hf_download_job, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def active_download():
    with _download_lock:
        active = [job for job in _download_jobs.values() if job.get("status") in ("queued", "downloading")]
        if not active:
            return {"job_id": None, "status": "idle"}
        active.sort(key=lambda job: job.get("started_at", 0), reverse=True)
        job = active[0]
        return {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "model_id": job.get("model_id"),
            "target_dir": job.get("target_dir"),
            "started_at": job.get("started_at"),
        }


def download_job_snapshot(job_id: str):
    with _download_lock:
        job = _download_jobs.get(job_id)
        if not job:
            return None
        return dict(job)

