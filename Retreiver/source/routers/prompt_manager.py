# source/routers/prompt_manager.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Literal, Callable

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

router = APIRouter(prefix="/prompt-manager", tags=["Prompt Manager"])

# -------- prefer factory; fallback to legacy singleton --------
_use_new_pm = False
try:
    from source.services.prompt_manager import get_prompt_manager  # type: ignore
    pm = get_prompt_manager()
    _use_new_pm = True
except Exception:
    from source.services.prompt_manager import prompt_manager as pm  # type: ignore
    _use_new_pm = False


# ---------------- Pydantic models (API-facing, backend-agnostic) ----------------

class SystemPromptUpsert(BaseModel):
    prompt_type: str = Field(..., description="Key/name for the system prompt")
    prompt: str = Field(..., description="Prompt content")
    overwrite: bool = Field(False, description="Replace if exists")

class TemplateIn(BaseModel):
    name: str
    template: str
    required_fields: List[str]
    optional_fields: List[str] = []
    description: str = ""
    version: str = "1.0"
    category: str = "general"
    max_length: int = 4000
    tags: List[str] = []

class TemplateOut(BaseModel):
    name: str
    description: str
    version: str
    category: str
    required_fields: List[str]
    optional_fields: List[str]
    tags: List[str]
    max_length: int

class TemplateBuildRequest(BaseModel):
    template_name: str
    fields: Dict[str, Any] = Field(default_factory=dict)  # arbitrary placeholders
    track_metrics: bool = True

class TemplateBuildResponse(BaseModel):
    prompt: str
    character_count: int
    estimated_tokens: int
    valid_length: bool

class RAGSource(BaseModel):
    content: Optional[str] = None
    text: Optional[str] = None
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None
    similarity_score: Optional[float] = None
    path: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None

class BuildRAGRequest(BaseModel):
    query: str
    sources: List[RAGSource] = Field(default_factory=list)
    system_prompt_type: str = "chat_with_rag"
    template_name: str = "chat_with_context"
    max_context_length: int = 3000

class BuildRAGResponse(BaseModel):
    system_prompt: str
    user_prompt: str

class ValidateRequest(BaseModel):
    prompt: str
    max_tokens: int = 4000

class ValidateResponse(BaseModel):
    character_count: int
    estimated_tokens: int
    line_count: int
    max_tokens: int
    is_valid: bool
    token_usage_ratio: float
    severity: str
    suggestions: List[str]

class OptimizeRequest(BaseModel):
    prompt: str
    target_tokens: int = 3000

class OptimizeResponse(BaseModel):
    optimized_prompt: str

class ImportPayload(BaseModel):
    templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    system_prompts: Dict[str, str] = Field(default_factory=dict)

class ExportPayload(BaseModel):
    templates: Dict[str, Dict[str, Any]]
    system_prompts: Dict[str, str]
    exported_at: str


# ---------------- helpers ----------------

def _http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)

async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
    res = fn(*args, **kwargs)
    if hasattr(res, "__await__"):
        return await res  # type: ignore
    return res

def _template_to_out(t: Any) -> TemplateOut:
    # tolerate dataclass/object/dict
    if isinstance(t, dict):
        data = t
    else:
        data = {
            "name": getattr(t, "name", None),
            "description": getattr(t, "description", ""),
            "version": getattr(t, "version", "1.0"),
            "category": getattr(t, "category", "general"),
            "required_fields": getattr(t, "required_fields", []) or [],
            "optional_fields": getattr(t, "optional_fields", []) or [],
            "tags": getattr(t, "tags", []) or [],
            "max_length": getattr(t, "max_length", 4000),
        }
    return TemplateOut(**data)

def _make_template_from_input(tpl: TemplateIn) -> Any:
    """
    Build a template object acceptable by the manager.
    - If manager exposes a dataclass/class `PromptTemplate`, use it.
    - Else, pass a dict; `add_template` should accept dicts too.
    """
    # try to find a class on the manager module
    PT = getattr(pm, "PromptTemplate", None)
    if PT:
        try:
            return PT(  # type: ignore
                name=tpl.name,
                template=tpl.template,
                required_fields=tpl.required_fields,
                optional_fields=tpl.optional_fields,
                description=tpl.description,
                version=tpl.version,
                category=tpl.category,
                max_length=tpl.max_length,
                tags=tpl.tags,
            )
        except Exception:
            # fall back to dict if constructor mismatches
            pass
    return {
        "name": tpl.name,
        "template": tpl.template,
        "required_fields": tpl.required_fields,
        "optional_fields": tpl.optional_fields,
        "description": tpl.description,
        "version": tpl.version,
        "category": tpl.category,
        "max_length": tpl.max_length,
        "tags": tpl.tags,
    }

def _default_max_tokens() -> int:
    # Prefer manager-provided default if available
    get_def = getattr(pm, "get_default_max_tokens", None)
    if callable(get_def):
        try:
            return int(get_def())
        except Exception:
            pass
    return 4000


# ---------------- basic routes ----------------

@router.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "prompt-manager"}

@router.get("/health")
async def health() -> Dict[str, Any]:
    hc = getattr(pm, "health_check", None)
    return await _maybe_await(hc) if callable(hc) else {"status": "ok"}


# ---------------- system prompts ----------------

@router.get("/system-prompts")
async def list_system_prompts() -> Dict[str, List[str]]:
    return {"prompts": sorted(await _maybe_await(pm.list_system_prompts))}

@router.get("/system-prompts/{prompt_type}")
async def get_system_prompt(prompt_type: str) -> Dict[str, str]:
    prompt = await _maybe_await(pm.get_system_prompt, prompt_type)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"System prompt '{prompt_type}' not found")
    return {"prompt_type": prompt_type, "prompt": prompt}

@router.get("/system-prompts/search")
async def search_system_prompts(q: str = Query(..., description="Keyword to search")) -> Dict[str, str]:
    return await _maybe_await(pm.search_system_prompts, q)

@router.post("/system-prompts/upsert")
async def upsert_system_prompt(req: SystemPromptUpsert) -> Dict[str, Any]:
    ok = await _maybe_await(pm.add_system_prompt, req.prompt_type, req.prompt, overwrite=req.overwrite)
    return {"updated": bool(ok), "prompt_type": req.prompt_type}


# ---------------- templates ----------------

@router.get("/templates")
async def list_templates(category: Optional[str] = Query(None)) -> Dict[str, Any]:
    names = await _maybe_await(pm.list_templates, category=category)
    return {"templates": names, "category": category}

@router.get("/templates/by-category")
async def get_templates_by_category() -> Dict[str, List[str]]:
    return await _maybe_await(pm.get_templates_by_category)

@router.get("/templates/{name}", response_model=TemplateOut)
async def get_template(name: str) -> TemplateOut:
    t = await _maybe_await(pm.get_template, name)
    if not t:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return _template_to_out(t)

@router.post("/templates/upsert")
async def upsert_template(tpl: TemplateIn, overwrite: bool = Query(False)) -> Dict[str, Any]:
    try:
        t_obj = _make_template_from_input(tpl)
        ok = await _maybe_await(pm.add_template, t_obj, overwrite=overwrite)
        return {"updated": bool(ok), "name": tpl.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upsert template: {e}")


# ---------------- prompt building ----------------

@router.post("/build", response_model=TemplateBuildResponse)
async def build_prompt(req: TemplateBuildRequest) -> TemplateBuildResponse:
    try:
        prompt = await _maybe_await(pm.build_prompt, req.template_name, track_metrics=req.track_metrics, **(req.fields or {}))
        # validate length
        max_tokens = _default_max_tokens()
        val = await _maybe_await(pm.validate_prompt_length, prompt, max_tokens=max_tokens)
        return TemplateBuildResponse(
            prompt=prompt,
            character_count=len(prompt),
            estimated_tokens=int(val.get("estimated_tokens", 0)),
            valid_length=bool(val.get("is_valid", True)),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Build failed: {e}")

@router.post("/build/rag", response_model=BuildRAGResponse)
async def build_rag(req: BuildRAGRequest) -> BuildRAGResponse:
    try:
        sources = [s.model_dump() for s in req.sources]
        builder = getattr(pm, "build_rag_prompt", None)
        if not callable(builder):
            raise _http_400("RAG prompt builder not available")
        system_prompt, user_prompt = await _maybe_await(
            builder,
            query=req.query,
            sources=sources,
            system_prompt_type=req.system_prompt_type,
            template_name=req.template_name,
            max_context_length=req.max_context_length,
        )
        return BuildRAGResponse(system_prompt=system_prompt, user_prompt=user_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"RAG prompt build failed: {e}")

@router.post("/build/analysis")
async def build_analysis_prompt(query: str = Body(..., embed=True), context: str = Body("", embed=True)) -> Dict[str, str]:
    try:
        prompt = await _maybe_await(pm.build_analysis_prompt, query, context=context)
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis prompt build failed: {e}")

@router.post("/build/planning")
async def build_planning_prompt(analysis: Dict[str, Any] = Body(..., embed=True)) -> Dict[str, str]:
    """
    Accepts the analysis object (as dict) your analyzer returns.
    """
    try:
        prompt = await _maybe_await(pm.build_planning_prompt, analysis)
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Planning prompt build failed: {e}")

@router.post("/build/synthesis")
async def build_synthesis_prompt(
    original_query: str = Body(..., embed=True),
    step_results: str = Body(..., embed=True),
    synthesis_instructions: str = Body("", embed=True),
) -> Dict[str, str]:
    try:
        prompt = await _maybe_await(
            pm.build_synthesis_prompt,
            original_query=original_query,
            step_results=step_results,
            synthesis_instructions=synthesis_instructions,
        )
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Synthesis prompt build failed: {e}")

@router.post("/build/summary")
async def build_document_summary_prompt(
    document_id: str = Body(..., embed=True),
    context: str = Body(..., embed=True),
    summary_type: str = Body("comprehensive", embed=True),
    target_length: str = Body("medium", embed=True),
    include_key_points: bool = Body(True, embed=True),
) -> Dict[str, str]:
    try:
        prompt = await _maybe_await(
            pm.build_document_summary_prompt,
            document_id=document_id,
            context=context,
            summary_type=summary_type,
            target_length=target_length,
            include_key_points=include_key_points,
        )
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Summary prompt build failed: {e}")

@router.post("/build/comparison")
async def build_comparison_prompt(
    topic: str = Body(..., embed=True),
    items: List[str] = Body(..., embed=True),
    context: str = Body("", embed=True),
    criteria: Optional[List[str]] = Body(None, embed=True),
) -> Dict[str, str]:
    try:
        prompt = await _maybe_await(
            pm.build_comparison_prompt,
            topic=topic,
            items=items,
            context=context,
            criteria=criteria or [],
        )
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison prompt build failed: {e}")


# ---------------- validation & optimization ----------------

@router.post("/validate", response_model=ValidateResponse)
async def validate_prompt(req: ValidateRequest) -> ValidateResponse:
    val = await _maybe_await(pm.validate_prompt_length, req.prompt, max_tokens=req.max_tokens)
    # accept extra fields but only expose API fields
    return ValidateResponse(
        character_count=int(val.get("character_count", len(req.prompt))),
        estimated_tokens=int(val.get("estimated_tokens", 0)),
        line_count=int(val.get("line_count", req.prompt.count("\n") + 1)),
        max_tokens=int(val.get("max_tokens", req.max_tokens)),
        is_valid=bool(val.get("is_valid", True)),
        token_usage_ratio=float(val.get("token_usage_ratio", 0.0)),
        severity=str(val.get("severity", "info")),
        suggestions=list(val.get("suggestions", [])),
    )

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(req: OptimizeRequest) -> OptimizeResponse:
    optimized = await _maybe_await(pm.optimize_prompt_length, req.prompt, target_tokens=req.target_tokens)
    return OptimizeResponse(optimized_prompt=str(optimized))


# ---------------- metrics & cache ----------------

@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    metrics = await _maybe_await(pm.get_all_metrics)
    out = {}
    for name, m in metrics.items():
        # tolerate dict or object
        if isinstance(m, dict):
            out[name] = {
                "usage_count": m.get("usage_count", 0),
                "avg_response_time": m.get("avg_response_time", 0),
                "avg_tokens_used": m.get("avg_tokens_used", 0),
                "success_rate": m.get("success_rate", 0),
                "last_used": m.get("last_used", None),
            }
        else:
            out[name] = {
                "usage_count": getattr(m, "usage_count", 0),
                "avg_response_time": getattr(m, "avg_response_time", 0),
                "avg_tokens_used": getattr(m, "avg_tokens_used", 0),
                "success_rate": getattr(m, "success_rate", 0),
                "last_used": getattr(m, "last_used", None),
            }
    return {"metrics": out}

@router.get("/metrics/{template_name}")
async def get_template_metrics(template_name: str) -> Dict[str, Any]:
    m = await _maybe_await(pm.get_template_metrics, template_name)
    if not m:
        raise HTTPException(status_code=404, detail="No metrics for template")
    if isinstance(m, dict):
        return {
            "template": template_name,
            "usage_count": m.get("usage_count", 0),
            "avg_response_time": m.get("avg_response_time", 0),
            "avg_tokens_used": m.get("avg_tokens_used", 0),
            "success_rate": m.get("success_rate", 0),
            "last_used": m.get("last_used", None),
        }
    return {
        "template": template_name,
        "usage_count": getattr(m, "usage_count", 0),
        "avg_response_time": getattr(m, "avg_response_time", 0),
        "avg_tokens_used": getattr(m, "avg_tokens_used", 0),
        "success_rate": getattr(m, "success_rate", 0),
        "last_used": getattr(m, "last_used", None),
    }

@router.post("/metrics/reset")
async def reset_metrics(template_name: Optional[str] = Body(None, embed=True)) -> Dict[str, str]:
    await _maybe_await(pm.reset_metrics, template_name)
    return {"status": "reset", "template_name": template_name or "*"}

@router.get("/cache")
async def cache_stats() -> Dict[str, Any]:
    getter = getattr(pm, "get_cache_stats", None)
    return await _maybe_await(getter) if callable(getter) else {"cache": "disabled"}

@router.post("/cache/clear")
async def cache_clear() -> Dict[str, Any]:
    clearer = getattr(pm, "clear_cache", None)
    cleared = await _maybe_await(clearer) if callable(clearer) else False
    return {"cleared": bool(cleared)}


# ---------------- import / export ----------------

@router.post("/export", response_model=ExportPayload)
async def export_templates() -> ExportPayload:
    # Export using manager methods; tolerate dict/object templates
    names = await _maybe_await(pm.list_templates)
    templates_dict: Dict[str, Dict[str, Any]] = {}
    for name in names:
        t = await _maybe_await(pm.get_template, name)
        if not t:
            continue
        if isinstance(t, dict):
            data = t
        else:
            data = {
                "name": getattr(t, "name", name),
                "template": getattr(t, "template", ""),
                "required_fields": getattr(t, "required_fields", []) or [],
                "optional_fields": getattr(t, "optional_fields", []) or [],
                "description": getattr(t, "description", ""),
                "version": getattr(t, "version", "1.0"),
                "category": getattr(t, "category", "general"),
                "tags": getattr(t, "tags", []) or [],
                "created_at": getattr(t, "created_at", None),
                "max_length": getattr(t, "max_length", 4000),
            }
        templates_dict[name] = data

    sys_prompts = {k: await _maybe_await(pm.get_system_prompt, k) for k in await _maybe_await(pm.list_system_prompts)}
    hc = getattr(pm, "health_check", None)
    exported_at = (await _maybe_await(hc)).get("timestamp", "") if callable(hc) else ""

    return ExportPayload(templates=templates_dict, system_prompts=sys_prompts, exported_at=exported_at)

@router.post("/import")
async def import_templates(payload: ImportPayload, overwrite: bool = Query(False)) -> Dict[str, int]:
    """
    Imports templates and system prompts from JSON payload.
    """
    imported_templates = 0
    imported_sys = 0

    # System prompts
    existing = set(await _maybe_await(pm.list_system_prompts))
    for name, prompt in payload.system_prompts.items():
        if name not in existing or overwrite:
            await _maybe_await(pm.add_system_prompt, name, prompt, overwrite=True)
            imported_sys += 1

    # Templates
    for name, data in payload.templates.items():
        # try strict init; fall back to tolerant mapping
        t_obj = {
            "name": data.get("name", name),
            "template": data.get("template", ""),
            "required_fields": data.get("required_fields", []),
            "optional_fields": data.get("optional_fields", []),
            "description": data.get("description", ""),
            "version": data.get("version", "1.0"),
            "category": data.get("category", "general"),
            "max_length": data.get("max_length", 4000),
            "tags": data.get("tags", []),
        }
        PT = getattr(pm, "PromptTemplate", None)
        if PT:
            try:
                t_obj = PT(**t_obj)  # type: ignore
            except Exception:
                # leave as dict if schema mismatch
                pass
        ok = await _maybe_await(pm.add_template, t_obj, overwrite=overwrite)
        if ok:
            imported_templates += 1

    return {"templates": imported_templates, "system_prompts": imported_sys}
