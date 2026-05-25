# source/core/prompt_manager_settings.py
# type: ignore
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# There isn't a dedicated loader for prompt_manager in source.config,
# so read the whole YAML once via the internal reader.
from source.config import reload_config as _read_all_config  # cached by our lru_cache


def _b(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return default

def _i(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


class TemplateSpec(BaseModel):
    name: str
    template: str
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    category: str = "general"
    max_length: int = 4000
    tags: List[str] = Field(default_factory=list)


class PromptManagerSettings(BaseModel):
    # Caching & metrics
    cache_enabled: bool = True
    cache_max_entries: int = Field(default=256, ge=0)
    cache_ttl_seconds: int = Field(default=900, ge=0)
    enable_metrics: bool = True

    # Token estimation
    chars_per_token: int = Field(default=4, ge=1)

    # Defaults used by routes/services
    default_system_prompt_type: str = "chat_with_rag"
    default_template_name: str = "chat_with_context"
    max_context_length: int = Field(default=3000, ge=1)

    # Seed content (optional; manager may import these at startup)
    system_prompts: Dict[str, str] = Field(default_factory=dict)
    templates: Dict[str, TemplateSpec] = Field(default_factory=dict)


def _normalize_templates(raw: Any) -> Dict[str, TemplateSpec]:
    """
    Accepts either:
      - mapping: {name: {template, required_fields, ...}}
      - list of items: [{name:..., template:...}, ...]
    Returns a dict[name] -> TemplateSpec
    """
    out: Dict[str, TemplateSpec] = {}
    if isinstance(raw, dict):
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                # treat value as the template string
                out[str(name)] = TemplateSpec(name=str(name), template=str(spec))
                continue
            out[str(name)] = TemplateSpec(
                name=str(spec.get("name") or name),
                template=str(spec.get("template", "")),
                required_fields=list(spec.get("required_fields") or []),
                optional_fields=list(spec.get("optional_fields") or []),
                description=str(spec.get("description", "")),
                version=str(spec.get("version", "1.0")),
                category=str(spec.get("category", "general")),
                max_length=_i(spec.get("max_length", 4000), 4000),
                tags=list(spec.get("tags") or []),
            )
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            out[str(item["name"])] = TemplateSpec(
                name=str(item["name"]),
                template=str(item.get("template", "")),
                required_fields=list(item.get("required_fields") or []),
                optional_fields=list(item.get("optional_fields") or []),
                description=str(item.get("description", "")),
                version=str(item.get("version", "1.0")),
                category=str(item.get("category", "general")),
                max_length=_i(item.get("max_length", 4000), 4000),
                tags=list(item.get("tags") or []),
            )
    return out


@lru_cache(maxsize=1)
def get_prompt_manager_settings() -> PromptManagerSettings:
    # Read the whole YAML once, then pluck "prompt_manager"
    cfg_all = _read_all_config() or {}
    cfg = cfg_all.get("prompt_manager") or {}

    return PromptManagerSettings(
        cache_enabled=_b(cfg.get("cache_enabled", True), True),
        cache_max_entries=_i(cfg.get("cache_max_entries", 256), 256),
        cache_ttl_seconds=_i(cfg.get("cache_ttl_seconds", 900), 900),
        enable_metrics=_b(cfg.get("enable_metrics", True), True),

        chars_per_token=_i(cfg.get("chars_per_token", 4), 4),

        default_system_prompt_type=str(cfg.get("default_system_prompt_type", "chat_with_rag")),
        default_template_name=str(cfg.get("default_template_name", "chat_with_context")),
        max_context_length=_i(cfg.get("max_context_length", 3000), 3000),

        system_prompts=dict(cfg.get("system_prompts") or {}),
        templates=_normalize_templates(cfg.get("templates")),
    )
