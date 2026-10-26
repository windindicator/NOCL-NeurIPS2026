"""Portable JSON and non-pickle Haiku checkpoints."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_params(path, params):
    flat = {f'{module}::{key}': np.asarray(value)
            for module, values in params.items() for key, value in values.items()}
    np.savez_compressed(path, **flat)


def load_params(path):
    params = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            module, key = name.rsplit('::', 1)
            params.setdefault(module, {})[key] = jnp.asarray(arrays[name])
    return params


def environment():
    distributions = ['jax', 'jaxlib', 'dm-haiku', 'optax', 'numpy', 'scipy',
                     'matplotlib']
    gpu_versions = {distribution.metadata['Name']: distribution.version
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata['Name'].startswith(('jax-cuda', 'nvidia-'))}
    sources = Path(__file__).parent
    return {
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in distributions},
        'gpu_packages': gpu_versions,
        'devices': [str(device) + ' ' + device.device_kind for device in jax.devices()],
        'jax_enable_x64': bool(jax.config.jax_enable_x64),
        'source_sha256': {
            str(path.relative_to(sources)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources.rglob('*.py'))},
    }
