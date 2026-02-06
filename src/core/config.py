import os
import re
from copy import deepcopy

import yaml

_REF_RE = re.compile(r"\$\{([^}]+)\}")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _get_by_path(root, path):
    current = root
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_refs(obj, root):
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    if isinstance(obj, str):
        def _replace(match):
            value = _get_by_path(root, match.group(1))
            return str(value) if value is not None else match.group(0)
        return _REF_RE.sub(_replace, obj)
    return obj


def load_config(path):
    config_path = os.path.abspath(path)
    config_dir = os.path.dirname(config_path)
    config = _load_yaml(config_path)
    base_path = config.get("base_config")
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)
        base_config = _load_yaml(base_path)
        config = deep_merge(base_config, {k: v for k, v in config.items() if k != "base_config"})
    config = _resolve_refs(config, config)
    return config
