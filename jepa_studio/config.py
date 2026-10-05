"""Shared run-config format (schema/config.schema.json) and a tiny dependency-free validator.

The browser app ships the same schema and an equivalent validator (site/js/config.js);
tests/test_web_engine.py (via tests/js/config_check.mjs) checks that both accept and reject
the same example files.
Supported keywords: type, enum, const, minimum, maximum, maxLength, pattern, required,
properties, additionalProperties(false), items, minItems, maxItems, oneOf.
"""
from __future__ import annotations

import re

import copy
import json
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "config.schema.json"


def load_schema() -> dict:
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text())
    return json.loads(resources.files("jepa_studio").joinpath("config.schema.json").read_text())


class ConfigError(ValueError):
    pass


def _type_ok(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    raise ConfigError(f"schema uses unsupported type {t}")


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errs: list[str] = []
    if "oneOf" in schema:
        matches = [s for s in schema["oneOf"] if not validate(value, s, path)]
        if len(matches) != 1:
            errs.append(f"{path}: must match exactly one of {json.dumps(schema['oneOf'])}")
        return errs
    if "const" in schema and value != schema["const"]:
        return [f"{path}: must be {schema['const']!r}"]
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(value, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        errs.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        errs.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing required key '{k}'")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unknown key '{k}'")
    return errs


DEFAULT_CONFIG: dict = {
    "version": 1,
    "name": "shapes-quickstart",
    "seed": 0,
    "data": {
        "kind": "synthetic-shapes", "path": None, "max_items": 4096,
        "image_size": 32, "local_size": 16, "channels": 3,
        "n_global": 2, "n_local": 4,
        "global_scale": [0.3, 1.0], "local_scale": [0.05, 0.3],
        "flip_p": 0.5, "color_jitter": 0.4, "grayscale_p": 0.2, "blur_p": 0.2,
        "mask_ratio": 0.0, "labeled_fraction": 0.1,
        "clip_frames": 8, "series_window": 128,
    },
    "model": {"arch": "convnet-tiny", "embed_dim": 128, "proj_dim": 16, "proj_hidden": 256,
              "patch_size": 4, "depth": 4, "heads": 4},
    "objective": {"lambda": 0.05, "num_slices": 256, "t_max": 3.0, "knots": 17},
    "train": {"epochs": 10, "max_steps": None, "batch_size": "auto", "effective_batch": 256,
              "lr": 5e-4, "weight_decay": 5e-2, "warmup_frac": 0.1, "final_lr_ratio": 1e-3,
              "checkpoint_every": 200, "log_every": 10},
    "hardware": {"mode": "balanced", "device": "auto", "precision": "auto", "channels_last": "auto",
                 "compile": False, "workers": "auto", "pin_memory": "auto",
                 "max_memory_frac": 0.85, "max_temp_c": 85, "max_disk_gb": 5},
    "eval": {"knn_k": 20, "probe_epochs": 100, "baselines": ["random-init", "supervised"]},
    "world": {"env": "two-room", "episodes": 400, "episode_len": 24, "history": 1, "horizon": 5,
              "cem_samples": 300, "cem_elites": 30, "cem_iters": 10},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
        if len(raw) > 1_000_000:
            raise ConfigError("config file larger than 1 MB; refusing to parse")
        cfg = deep_merge(cfg, json.loads(raw))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    errs = validate(cfg, load_schema())
    if errs:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errs))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    errs = validate(cfg, load_schema())
    if errs:
        raise ConfigError("refusing to save an invalid config:\n  " + "\n  ".join(errs))
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
