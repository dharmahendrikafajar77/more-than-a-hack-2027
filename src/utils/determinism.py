"""Determinism and reproducibility utilities for More Than a Hack 2027."""

import os
import random
import hashlib
import json
import datetime
import numpy as np

DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    """Lock random seeds for reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_file_sha256(file_path):
    """Compute standard SHA-256 hex digest of any file."""
    if not os.path.isfile(file_path):
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_manifest(manifest_path, config_dict, output_files=None, extra_metadata=None):
    """Record execution manifest for full reproducibility auditing."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    files_hashes = {}
    if output_files:
        for name, path in output_files.items():
            files_hashes[name] = {
                "path": path,
                "sha256": compute_file_sha256(path),
                "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0
            }
            
    manifest = {
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "config": config_dict,
        "outputs": files_hashes,
        "metadata": extra_metadata or {}
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest

