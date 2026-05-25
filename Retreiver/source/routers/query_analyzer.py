# source/routers/query_analyzer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Callable, Coroutine

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

router = APIRouter(prefix="/query", tags=["Query Analyzer"])

# ---- Try a factory-based analyzer; fall back to legacy service singleton ----
_use_new_analyzer = False
try:
    # If you add a factory later (e.g., to swap LLMs/prompts via config),
    # expose it as source.services.query_analyzer_service.get_query_analyzer
    from source.services.query_analyzer_service import get_query_analyzer  # type: ignore
    _qa = get_query_analyzer()
    _use_new_analyzer = True
except Exception:
    from source.services.query_analyzer import query_analyzer  # type: ignore
    _qa = query_analyzer
    _use_new_analyzer = False


# ----------------------------- Pydantic I/O models ----------------------------- #

IntentLiteral = Literal[
    "factual", "analytical", "procedural", "creative", "research", "clarification", "conversational"
]
ComplexityLiteral = Literal["simple", "moderate", "complex", "research_intensive"]
ContextLiteral = Literal["none", "minimal", "moderate", "extensive"]


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's natural language query")


class QueryAnalysisModel(BaseModel):
    """Pure API model — no dependency on service dataclasses."""
    original_query: str
    intent: IntentLiteral
    complexity: ComplexityLiteral
    context_requirement: ContextLiteral
    key_concepts: List[str]
    sub_questions: List[str]
    search_terms: List[str]
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    estimated_tokens: int = Field(ge=0)
    requires_multi_step: bool
    domain_hints: List[str] = []
    query_type: str
    ambiguity_score: float = Field(ge=0, le=1)
    timestamp: str

    @classmethod
    def from_service_like(cls, obj: Any) -> "QueryAnalysisModel":
        """
        Convert various service return shapes to API model:
        - obj.to_dict()
        - dataclasses.asdict(obj)
        - obj.dict()
        - obj.__dict__
        - already-a-dict
        """
        if isinstance(obj, dict):
            data = obj
        else:
            # try .to_dict()
            to_dict = getattr(obj, "to_dict", None)
            if callable(to_dict):
                data = to_dict()
            else:
                # try dataclasses.asdict
                try:
                    from dataclasses import asdict, is_dataclass
                    if is_dataclass(obj):
                        data = asdict(obj)
                    else:
                        raise TypeError
                except Exception:
                    # try .dict()
                    dmethod = getattr(obj, "dict", None)
                    if callable(dmethod):
                        data = dmethod()
                    else:
                        # last resort: __dict__
                        data = getattr(obj, "__dict__", {})
                        if not isinstance(data, dict):
                            raise TypeError("Unsupported analysis object type")
        return cls(**data)


class SearchStrategyRequest(BaseModel):
    query: Optional[str] = Field(None)
    analysis: Optional[QueryAnalysisModel] = Field(None)

    @model_validator(mode="after")
    def ensure_one_input(self):
        if not self.analysis and not self.query:
            raise ValueError("Provide either 'query' or 'analysis'.")
        return self



class SearchStrategyResponse(BaseModel):
    primary_terms: List[str]
    secondary_terms: List[str]
    search_type: Literal["semantic"]  # keep existing API contract
    source_count: int
    search_scope: Literal["focused", "broad"]
    filters: Dict[str, Any]


class ValidateRequest(BaseModel):
    analysis: QueryAnalysisModel
    actual_tokens: Optional[int] = Field(None, ge=0)
    actual_quality: Optional[float] = Field(None, ge=0, le=1)


# ----------------------------- small helpers ----------------------------- #

def _http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)

def _http_500(msg: str) -> HTTPException:
    return HTTPException(status_code=500, detail=msg)

async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call a function that may be sync or async.
    """
    res = fn(*args, **kwargs)
    if hasattr(res, "__await__"):
        return await res  # type: ignore
    return res


# ----------------------------- Routes ----------------------------- #

@router.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "query-analyzer"}


@router.post("/analyze", response_model=QueryAnalysisModel)
async def analyze(req: AnalyzeRequest) -> QueryAnalysisModel:
    try:
        if not req.query or not req.query.strip():
            raise _http_400("Query cannot be empty")

        # Prefer `analyze_query(query: str)`; tolerate alternative naming
        analyze_fn = getattr(_qa, "analyze_query", None) or getattr(_qa, "analyze", None)
        if not callable(analyze_fn):
            raise _http_500("Analyzer service missing 'analyze_query' method")

        service_obj = await _maybe_await(analyze_fn, req.query)
        return QueryAnalysisModel.from_service_like(service_obj)
    except HTTPException:
        raise
    except ValueError as ve:
        raise _http_400(str(ve))
    except Exception as e:
        raise _http_500(f"Analyzer error: {e}")


@router.post("/search-strategy", response_model=SearchStrategyResponse)
async def search_strategy(req: SearchStrategyRequest) -> SearchStrategyResponse:
    try:
        # Build or reuse analysis
        if req.analysis:
            analysis_payload = req.analysis.dict()
        else:
            analyze_fn = getattr(_qa, "analyze_query", None) or getattr(_qa, "analyze", None)
            if not callable(analyze_fn):
                raise _http_500("Analyzer service missing 'analyze_query' method")
            service_obj = await _maybe_await(analyze_fn, req.query or "")
            analysis_payload = QueryAnalysisModel.from_service_like(service_obj).dict()

        # Prefer `get_search_strategy(analysis_like)` and accept dict
        strategy_fn = getattr(_qa, "get_search_strategy", None)
        if not callable(strategy_fn):
            raise _http_500("Analyzer service missing 'get_search_strategy' method")

        strategy = await _maybe_await(strategy_fn, analysis_payload)
        # Normalize keys in case service returns extra fields
        return SearchStrategyResponse(**{
            "primary_terms": strategy.get("primary_terms", []),
            "secondary_terms": strategy.get("secondary_terms", []),
            "search_type": strategy.get("search_type", "semantic"),
            "source_count": strategy.get("source_count", 5),
            "search_scope": strategy.get("search_scope", "focused"),
            "filters": strategy.get("filters", {}) or {},
        })
    except HTTPException:
        raise
    except ValueError as ve:
        raise _http_400(str(ve))
    except Exception as e:
        raise _http_500(f"Strategy error: {e}")


@router.post("/validate")
async def validate(req: ValidateRequest) -> Dict[str, Any]:
    try:
        validate_fn = getattr(_qa, "validate_analysis", None)
        if not callable(validate_fn):
            raise _http_500("Analyzer service missing 'validate_analysis' method")

        result = await _maybe_await(
            validate_fn,
            req.analysis.dict(),
            actual_tokens=req.actual_tokens,
            actual_quality=req.actual_quality,
        )
        # Ensure JSON-serializable dict
        return dict(result) if not isinstance(result, dict) else result
    except HTTPException:
        raise
    except Exception as e:
        raise _http_500(f"Validation error: {e}")
