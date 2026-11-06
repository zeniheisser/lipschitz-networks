# Limits how much of the CPU the notebooks use, runs independent jobs (network trainings, floors) side by side, and
# stores their results on disk, so that a notebook only computes what it has not computed before.
#
# Settings, either as environment variables before starting the kernel, or in the notebook:
#     LIPSCHITZ_MAX_JOBS=4   or  job_runner.set_max_jobs(4)   the most processes or threads to use (default 8)
#     LIPSCHITZ_CACHE=0      or  job_runner.USE_CACHE = False always compute again, and store nothing
#     LIPSCHITZ_CACHE_DIR=.. where the results are stored (default: run_cache/ next to this file)
# Deleting the run_cache folder deletes all stored results. Nothing else depends on it.
#
# A stored result belongs to a hash of everything that goes into it: the job (data, seed, steps, ...) and the source code
# of the functions and classes that compute it. Changing any of those computes the result again. The old result stays
# on disk, and is used again if the change is undone.
import hashlib
import inspect
import multiprocessing as mp
import numbers
import os
import tempfile
import time

import numpy as np
import scipy.linalg  # noqa: F401, scipy has its own BLAS, which has to be loaded for threadpool_limits to reach it
import torch
from threadpoolctl import threadpool_limits

MAX_JOBS = int(os.environ.get("LIPSCHITZ_MAX_JOBS", 8))
USE_CACHE = os.environ.get("LIPSCHITZ_CACHE", "1") != "0"
CACHE_DIR = os.environ.get("LIPSCHITZ_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_cache"))


##### LIMITING THE CPU USE #####
def _limit_threads(n):
    """At most n threads per process for torch and for numpy and scipy (BLAS), which otherwise use every core."""
    torch.set_num_threads(n)
    threadpool_limits(n)


def set_max_jobs(n):
    """Changes the limit for the running kernel: the number of jobs side by side, and the threads of a single job."""
    global MAX_JOBS
    MAX_JOBS = n
    _limit_threads(n)


_limit_threads(MAX_JOBS)


##### STORING RESULTS #####
def _hash_into(h, obj):
    """Adds obj to the hash h. Raises TypeError for objects whose content cannot be determined."""
    if obj is None or isinstance(obj, (bool, str, bytes)):
        h.update(repr(obj).encode())
    elif isinstance(obj, numbers.Real):  # also numpy numbers. 1, 1.0 and np.float64(1) are the same setting
        h.update(float(obj).hex().encode())
    elif isinstance(obj, torch.Tensor):
        _hash_into(h, obj.detach().cpu().numpy())
    elif isinstance(obj, np.ndarray):
        h.update(f"{obj.dtype}{obj.shape}".encode())
        h.update(obj.tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            _hash_into(h, str(key))
            _hash_into(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(f"{len(obj)}".encode())
        for item in obj:
            _hash_into(h, item)
    elif inspect.isclass(obj) or inspect.isroutine(obj):
        h.update(obj.__qualname__.encode())
        try:
            h.update(inspect.getsource(obj).encode())
        except (OSError, TypeError):  # built in, so there is no source
            pass
    elif isinstance(obj, torch.nn.Module):
        _hash_into(h, type(obj))
        h.update(repr(obj).encode())
        _hash_into(h, dict(obj.state_dict()))
        # Settings which are not in the repr or the weights, like the Lipschitz bound
        _hash_into(h, {k: v for k, v in vars(obj).items() if not k.startswith("_") and isinstance(v, (numbers.Real, str))})
    elif isinstance(obj, torch.Generator):
        _hash_into(h, obj.get_state())
    elif isinstance(obj, torch.utils.data.Dataset):
        _hash_into(h, type(obj))
        _hash_into(h, vars(obj))
    else:
        raise TypeError(f"Cannot hash an object of type {type(obj).__name__}")


def cache_key(*parts):
    """Hash of everything that goes into a result."""
    h = hashlib.blake2b(digest_size=16)
    _hash_into(h, parts)
    return h.hexdigest()


def load_result(key):
    """The stored result with this key, or None if there is none."""
    path = os.path.join(CACHE_DIR, f"{key}.pt")
    if not USE_CACHE or not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # e.g. written by an incompatible version: compute it again
        print(f"Could not read the stored result {path} ({error}), computing it again")
        return None


def save_result(key, result):
    """Stores a result. Written to a temporary file first, so that an interrupted run cannot leave half a file."""
    if not USE_CACHE:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(result, temporary_path)
        os.replace(temporary_path, os.path.join(CACHE_DIR, f"{key}.pt"))
    except BaseException:
        os.remove(temporary_path)
        raise


##### RUNNING JOBS #####
def _init_worker():
    _limit_threads(1)  # one core per job


def _run_job(call):
    key, function, job = call
    return key, function(job)


def run_jobs(function, jobs, depends_on=(), n_workers=None, label="Jobs"):
    """
    Runs function(job) for every job and returns the results in the same order, at most MAX_JOBS at a time.

    Stored results are loaded, identical jobs are computed once, and only the rest is computed, one process per job.
    Every result is stored as soon as it is finished, so an interrupted run continues where it stopped.

    function must be importable (defined in a module, not in the notebook) to be sent to the worker processes.
    depends_on: the other functions and classes the result depends on, whose source code is part of the hash.
    n_workers: fewer processes than MAX_JOBS, if wanted.
    """
    code = (function, tuple(depends_on))
    keys = [cache_key(code, job) for job in jobs]

    results = {}
    missing = {}  # one job per key that is not stored yet
    for key, job in zip(keys, jobs):
        if key in results or key in missing:
            continue
        result = load_result(key)
        if result is None:
            missing[key] = job
        else:
            results[key] = result

    if missing:
        n_workers = min(n_workers or MAX_JOBS, MAX_JOBS, len(missing))
        print(f"{label}: computing {len(missing)} of {len(jobs)} with {n_workers} processes")
        start = time.perf_counter()
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker) as pool:
            calls = [(key, function, job) for key, job in missing.items()]
            for done, (key, result) in enumerate(pool.imap_unordered(_run_job, calls), start=1):
                save_result(key, result)
                results[key] = result
                print(f"\r{label}: {done}/{len(calls)} done after {(time.perf_counter() - start) / 60:.1f} min", end="", flush=True)
        print()
    else:
        print(f"{label}: all {len(jobs)} results already available")
    return [results[key] for key in keys]
