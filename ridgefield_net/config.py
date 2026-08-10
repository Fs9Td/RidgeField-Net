import copy
import json
from pathlib import Path

import yaml


class ConfigNode(dict):
    """支持属性访问的轻量级 YAML 配置对象。"""

    def __init__(self, value=None):
        super().__init__()
        for key, item in (value or {}).items():
            self[key] = self._wrap(item)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(x) for x in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def to_dict(self):
        out = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [x.to_dict() if isinstance(x, ConfigNode) else x for x in value]
            else:
                out[key] = value
        return out

    def clone(self):
        return ConfigNode(copy.deepcopy(self.to_dict()))

    def pretty(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _parse_value(raw):
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except Exception:
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            return raw


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = ConfigNode(yaml.safe_load(handle))
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"配置覆盖必须是 key=value：{override}")
        key, raw = override.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], ConfigNode):
                node[part] = ConfigNode()
            node = node[part]
        node[parts[-1]] = ConfigNode._wrap(_parse_value(raw))
    cfg._config_path = str(Path(path).resolve())
    return cfg
