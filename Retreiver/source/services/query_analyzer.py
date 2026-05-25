# source/services/query_analyzer.py
# type: ignore
from __future__ import annotations

"""
Query Analyzer Service
----------------------
Analyzes user queries to determine intent, complexity, and an optimal search/response strategy.

Design goals:
- Single LLM call site via llm_client.create_completion(...)
- Prompts referenced by keys from config.yaml (via load_query_settings) and resolved by prompt_manager
- Optional heuristic-only mode (no LLM dependency)
- Async-friendly (wraps sync client calls), robust JSON parsing, retry/backoff
- Tolerant I/O: accepts/returns plain dicts for easy use by routers

Config sketch (config.yaml):

query_settings:
  analysis_model: "gpt-4o"
  temperature_default: 0.1
  max_tokens_default: 800
  heuristic_only: false
  llm_retries: 2
  llm_retry_base_delay: 0.6
  llm_retry_jitter: 0.2
  prompt_keys:
    system_prompt: "query_analysis"
    template_name: "query_analysis_comprehensive"
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from source.config import load_query_settings
from source.services.llm_client import llm_client
from source.services.prompt_manager import prompt_manager

# ----------------------------- Config ----------------------------- #

query_settings = load_query_settings() or {}

_PROMPT_KEYS: Dict[str, str] = query_settings.get("prompt_keys") or {}
_SYSTEM_PROMPT_KEY: str = _PROMPT_KEYS.get("system_prompt", "query_analysis")
_TEMPLATE_NAME_KEY: str = _PROMPT_KEYS.get("template_name", "query_analysis_comprehensive")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ----------------------------- Enums & Models ----------------------------- #

class QueryIntent(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"
    RESEARCH = "research"
    CLARIFICATION = "clarification"
    CONVERSATIONAL = "conversational"


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH_INTENSIVE = "research_intensive"


class ContextRequirement(Enum):
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"


@dataclass
class QueryAnalysis:
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    context_requirement: ContextRequirement
    key_concepts: List[str]
    sub_questions: List[str]
    search_terms: List[str]
    confidence: float
    reasoning: str
    estimated_tokens: int
    requires_multi_step: bool
    domain_hints: List[str]
    query_type: str
    ambiguity_score: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intent"] = self.intent.value
        d["complexity"] = self.complexity.value
        d["context_requirement"] = self.context_requirement.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysis":
        return cls(
            original_query=data["original_query"],
            intent=QueryIntent(data["intent"]),
            complexity=QueryComplexity(data["complexity"]),
            context_requirement=ContextRequirement(data["context_requirement"]),
            key_concepts=list(data.get("key_concepts", [])),
            sub_questions=list(data.get("sub_questions", [])),
            search_terms=list(data.get("search_terms", [])),
            confidence=float(data.get("confidence", 0.7)),
            reasoning=str(data.get("reasoning", "")),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            requires_multi_step=bool(data.get("requires_multi_step", False)),
            domain_hints=list(data.get("domain_hints", [])),
            query_type=str(data.get("query_type", "question")),
            ambiguity_score=float(data.get("ambiguity_score", 0.0)),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
        )


# ----------------------------- Utilities ----------------------------- #

async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Call a function that may be sync or async."""
    try:
        res = fn(*args, **kwargs)
    except TypeError:
        # Some builders might be defined with positional vs keyword args: retry with kwargs only
        res = fn(**kwargs)
    if hasattr(res, "__await__"):
        return await res  # type: ignore
    return res


def _coalesce(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is not None:
            return v
    return default


# ----------------------------- Query Analyzer ----------------------------- #

class QueryAnalyzer:
    """Analyzes queries to determine characteristics and optimal handling strategy."""

    def __init__(
        self,
        retry_policy: Optional[Dict[str, Any]] = None,
        *,
        analysis_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        heuristic_only: Optional[bool] = None,
        system_prompt_key: Optional[str] = None,
        template_name_key: Optional[str] = None,
    ):
        # Generation params
        self.analysis_model: str = analysis_model or query_settings.get("analysis_model", "gpt-4o")
        self.temperature: float = float(_coalesce(temperature, query_settings.get("temperature_default"), 0.1))
        self.max_tokens: int = int(_coalesce(max_tokens, query_settings.get("max_tokens_default"), 800))
        self.heuristic_only: bool = bool(_coalesce(heuristic_only, query_settings.get("heuristic_only"), False))

        # Prompt keys
        self.system_prompt_key: str = system_prompt_key or _SYSTEM_PROMPT_KEY
        self.template_name_key: str = template_name_key or _TEMPLATE_NAME_KEY

        # Retry/backoff policy
        self.retry_policy = {
            "retries": int(query_settings.get("llm_retries", 2)),
            "base_delay": float(query_settings.get("llm_retry_base_delay", 0.6)),
            "jitter": float(query_settings.get("llm_retry_jitter", 0.2)),
        }
        if retry_policy:
            self.retry_policy.update(retry_policy)

        # Heuristics configuration
        self.question_indicators = query_settings.get(
            "question_indicators",
            [
                "what", "how", "why", "when", "where", "who", "which", "whose",
                "can", "could", "would", "should", "will", "do", "does", "did",
                "is", "are", "was", "were",
            ],
        )
        self.complexity_indicators = query_settings.get(
            "complexity_indicators",
            {
                "simple": ["define", "what is", "who is", "when did"],
                "moderate": ["explain", "describe", "compare", "list"],
                "complex": ["analyze", "evaluate", "synthesize", "develop", "design"],
                "research": ["comprehensive", "all aspects", "in-depth", "thorough analysis"],
            },
        )

    # ----------------------------- Public API ----------------------------- #

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Main entry: returns a QueryAnalysis dataclass."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        heuristic = self._heuristic_analysis(query)

        if self.heuristic_only:
            return heuristic

        try:
            ai_data = await self._ai_analysis(query)
            return self._merge_analyses(heuristic, ai_data, query)
        except Exception as e:
            logger.warning(f"AI analysis failed; using heuristic only: {e}")
            return heuristic

    # Accepts dict-like or QueryAnalysis (router passes dict)
    async def get_search_strategy(self, analysis_like: Union[QueryAnalysis, Dict[str, Any]]) -> Dict[str, Any]:
        a = self._as_analysis(analysis_like)
        strategy = {
            "primary_terms": list(a.search_terms[:3]),
            "secondary_terms": list(a.key_concepts),
            "search_type": "semantic",  # keep API contract
            "source_count": self._get_optimal_source_count(a),
            "search_scope": "broad" if a.complexity == QueryComplexity.RESEARCH_INTENSIVE else "focused",
            "filters": {
                "domains": list(a.domain_hints),
                "recency": "recent" if "current" in a.original_query.lower() else "any",
            },
        }
        return strategy

    # Accepts dict-like or QueryAnalysis (router passes dict)
    async def validate_analysis(
        self,
        analysis_like: Union[QueryAnalysis, Dict[str, Any]],
        actual_tokens: Optional[int] = None,
        actual_quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        a = self._as_analysis(analysis_like)
        out: Dict[str, Any] = {
            "analysis_accuracy": 0.0,
            "prediction_errors": [],
            "improvement_suggestions": [],
        }

        if actual_tokens is not None and a.estimated_tokens > 0:
            token_acc = 1.0 - abs(a.estimated_tokens - actual_tokens) / max(a.estimated_tokens, actual_tokens)
            out["token_prediction_accuracy"] = max(0.0, min(1.0, token_acc))

        if actual_quality is not None:
            if a.confidence > 0.8 and actual_quality < 0.6:
                out["prediction_errors"].append("Overconfident prediction")
            elif a.confidence < 0.5 and actual_quality > 0.8:
                out["prediction_errors"].append("Underconfident prediction")

        return out

    # ----------------------------- Heuristics ----------------------------- #

    def _heuristic_analysis(self, query: str) -> QueryAnalysis:
        q = query.strip()
        ql = q.lower()
        words = ql.split()
        word_count = len(words)

        # Query type
        query_type = "question"
        if any(p in ql for p in ["please", "can you", "help me"]):
            query_type = "request"
        if any(p in ql for p in ["create", "generate", "write", "make", "build"]):
            query_type = "command"

        # Intent
        intent = QueryIntent.CONVERSATIONAL
        if any(w in words for w in self.question_indicators):
            intent = QueryIntent.FACTUAL
        if any(p in ql for p in ["compare", "analyze", "evaluate", "versus", "vs"]):
            intent = QueryIntent.ANALYTICAL
        if any(p in ql for p in ["how to", "steps", "process", "procedure"]):
            intent = QueryIntent.PROCEDURAL
        if any(p in ql for p in ["create", "write", "generate", "compose", "draft"]):
            intent = QueryIntent.CREATIVE

        # Complexity
        complexity = QueryComplexity.SIMPLE
        if word_count > 15 or " and " in ql or "," in ql:
            complexity = QueryComplexity.MODERATE
        if any(p in ql for p in self.complexity_indicators.get("complex", [])):
            complexity = QueryComplexity.COMPLEX
        if any(p in ql for p in self.complexity_indicators.get("research", [])):
            complexity = QueryComplexity.RESEARCH_INTENSIVE

        # Context requirement
        if intent == QueryIntent.CONVERSATIONAL:
            context_req = ContextRequirement.NONE
        elif complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH_INTENSIVE):
            context_req = ContextRequirement.EXTENSIVE
        elif intent == QueryIntent.ANALYTICAL:
            context_req = ContextRequirement.MODERATE
        else:
            context_req = ContextRequirement.MINIMAL

        # Key concepts (very light, stop-wordy)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can",
        }
        key_concepts = [w.strip(".,!?()[]{}:;\"'") for w in words if len(w) > 3 and w not in stop_words][:5]

        # Ambiguity indicator
        ambiguity_indicators = ["maybe", "possibly", "might", "could", "various", "different", "some", "depends"]
        ambiguity_score = min(1.0, sum(1 for w in ambiguity_indicators if w in words) * 0.25)

        return QueryAnalysis(
            original_query=q,
            intent=intent,
            complexity=complexity,
            context_requirement=context_req,
            key_concepts=key_concepts,
            sub_questions=[q],
            search_terms=key_concepts[:3] or [q],
            confidence=0.6,
            reasoning="Heuristic analysis based on keyword patterns and length.",
            estimated_tokens=word_count * 3,
            requires_multi_step=(complexity == QueryComplexity.RESEARCH_INTENSIVE),
            domain_hints=self._extract_domain_hints(q),
            query_type=query_type,
            ambiguity_score=ambiguity_score,
            timestamp=datetime.now().isoformat(),
        )

    # ----------------------------- LLM Analysis (single call site) ----------------------------- #

    async def _ai_analysis(self, query: str) -> Dict[str, Any]:
        """
        AI-powered analysis using prompt_manager with keys from config.yaml.
        - System prompt: prompt_manager.get_system_prompt(self.system_prompt_key)
        - User prompt: prompt_manager.build_prompt(self.template_name_key, query=..., context=...)
        - LLM call: llm_client.create_completion(...) only
        """
        model = llm_client.resolve_model(self.analysis_model)

        # System/user prompt (tolerate sync/async prompt_manager)
        system_prompt = await _maybe_await(prompt_manager.get_system_prompt, self.system_prompt_key)  # type: ignore
        default_context = query_settings.get("default_context", "No additional context provided.")

        try:
            analysis_prompt = await _maybe_await(  # type: ignore
                prompt_manager.build_prompt,
                self.template_name_key,
                query=query,
                context=default_context,
            )
        except Exception:
            # Fallback to dedicated builder (if available)
            try:
                analysis_prompt = await _maybe_await(prompt_manager.build_analysis_prompt, query, context=default_context)  # type: ignore
            except Exception:
                analysis_prompt = f'Analyze this query and return ONLY JSON:\n\nQuery: "{query}"'

        # Single call site (wrap sync client in thread)
        def _call() -> str:
            resp = llm_client.create_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
                response_format=query_settings.get("response_format", {"type": "json_object"}),
            )
            try:
                return (resp.choices[0].message.content or "{}").strip()  # OpenAI-like
            except Exception:
                return "{}"

        raw = await self._with_retries(_call)
        return self._safe_parse_json(raw)

    # ----------------------------- Retry/Backoff ----------------------------- #

    async def _with_retries(self, fn: Callable[[], Any]) -> Any:
        retries = max(0, int(self.retry_policy.get("retries", 2)))
        base = float(self.retry_policy.get("base_delay", 0.6))
        jitter = float(self.retry_policy.get("jitter", 0.2))
        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                # fn is sync; run in thread
                return await asyncio.to_thread(fn)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    delay = base * (2 ** attempt) + random.uniform(0, jitter)
                    logger.warning(
                        "LLM analysis failed (attempt %d/%d). Retrying in %.2fs: %s",
                        attempt + 1, retries, delay, e,
                    )
                    await asyncio.sleep(delay)

        logger.error("LLM analysis failed after %d retries: %s", retries, last_err)
        raise last_err or RuntimeError("Query analysis retry exhaustion")

    # ----------------------------- JSON Parsing ----------------------------- #

    def _safe_parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON defensively; if text contains JSON, extract the innermost object."""
        # direct
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # extract largest {...}
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            try:
                parsed = json.loads(snippet)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                logger.debug("Snippet JSON parse failed")

        logger.warning("Failed to parse analysis JSON; raw (truncated): %r", content[:200])
        return {}

    # ----------------------------- Merge ----------------------------- #

    def _merge_analyses(self, heuristic: QueryAnalysis, ai: Dict[str, Any], query: str) -> QueryAnalysis:
        try:
            # Confidence clamped and optionally boosted by config
            base_conf = float(ai.get("confidence", heuristic.confidence))
            max_conf = float(query_settings.get("max_confidence", 0.95))
            boost = float(query_settings.get("confidence_boost", 0.0))
            conf = min(max_conf, base_conf + boost)

            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent(ai.get("intent", heuristic.intent.value)),
                complexity=QueryComplexity(ai.get("complexity", heuristic.complexity.value)),
                context_requirement=ContextRequirement(ai.get("context_requirement", heuristic.context_requirement.value)),
                key_concepts=list(ai.get("key_concepts", heuristic.key_concepts)),
                sub_questions=list(ai.get("sub_questions", heuristic.sub_questions)),
                search_terms=list(ai.get("search_terms", heuristic.search_terms)),
                confidence=conf,
                reasoning=str(ai.get("reasoning", heuristic.reasoning)),
                estimated_tokens=int(ai.get("estimated_tokens", heuristic.estimated_tokens)),
                requires_multi_step=bool(ai.get("requires_multi_step", heuristic.requires_multi_step)),
                domain_hints=list(ai.get("domain_hints", heuristic.domain_hints)),
                query_type=str(ai.get("query_type", heuristic.query_type)),
                ambiguity_score=float(ai.get("ambiguity_score", heuristic.ambiguity_score)),
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning("Error merging analyses, falling back to heuristic: %s", e)
            return heuristic

    # ----------------------------- Helpers ----------------------------- #

    def _extract_domain_hints(self, query: str) -> List[str]:
        # Defaults are intentionally broad; override via config if desired
        default_domain_keywords = {
            "technology": ["computer", "software", "ai", "machine learning", "programming", "algorithm", "data"],
            "science": ["biology", "chemistry", "physics", "research", "experiment", "hypothesis"],
            "business": ["market", "finance", "economics", "strategy", "management", "profit"],
            "education": ["learning", "teaching", "school", "university", "student", "curriculum"],
            "environment": ["climate", "environment", "sustainability", "renewable", "pollution"],
            "travel": [
                "flight", "flights", "airline", "pnr", "itinerary", "hotel", "booking", "reservation",
                "check-in", "boarding", "visa", "tourism", "trip", "fare", "baggage", "airport", "train", "bus",
            ],
            "healthcare": [
                "patient", "clinical", "hospital", "doctor", "diagnosis", "treatment", "therapy", "medical",
                "medication", "pharmacy", "laboratory", "radiology", "hipaa", "ehr", "claims",
            ],
            "insurance": [
                "policy", "premium", "claim", "underwriting", "coverage", "deductible", "copay", "coinsurance",
                "endorsement", "renewal", "lapse", "actuarial", "risk",
            ],
            "health": ["medicine", "health", "disease", "treatment", "medical", "patient"],  # back-compat
        }
        domain_keywords = query_settings.get("domain_keywords", default_domain_keywords)
        max_domains = int(query_settings.get("max_domain_hints", 3))

        q = query.lower()
        found = [d for d, kws in domain_keywords.items() if any(k in q for k in kws)]
        return found[:max_domains]

    def _get_optimal_source_count(self, a: QueryAnalysis) -> int:
        mapping = query_settings.get(
            "source_count_mapping",
            {"none": 0, "minimal": 2, "moderate": 5, "extensive": 8},
        )
        # keys may be stored upper-cased in some configs; normalize
        norm_map = {str(k).lower(): int(v) for k, v in mapping.items()}
        base = norm_map.get(a.context_requirement.value.lower(), int(query_settings.get("default_source_count", 3)))

        # Adjust by complexity
        if a.complexity == QueryComplexity.RESEARCH_INTENSIVE:
            base += int(query_settings.get("research_source_bonus", 2))
        elif a.complexity == QueryComplexity.SIMPLE:
            base = max(1, base - int(query_settings.get("simple_source_reduction", 1)))

        return min(int(query_settings.get("max_source_count", 10)), base)

    @staticmethod
    def _as_analysis(obj: Union[QueryAnalysis, Dict[str, Any]]) -> QueryAnalysis:
        if isinstance(obj, QueryAnalysis):
            return obj
        if isinstance(obj, dict):
            # Ensure required fields exist; if not, raise helpful error
            required = {
                "original_query", "intent", "complexity", "context_requirement",
                "key_concepts", "sub_questions", "search_terms", "confidence",
                "reasoning", "estimated_tokens", "requires_multi_step",
                "domain_hints", "query_type", "ambiguity_score", "timestamp",
            }
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"Analysis dict missing fields: {', '.join(missing)}")
            return QueryAnalysis.from_dict(obj)
        raise TypeError("analysis must be a QueryAnalysis or dict")


# ----------------------------- Factory & Singleton ----------------------------- #

def get_query_analyzer(**overrides: Any) -> QueryAnalyzer:
    """Factory for DI/tests; accepts constructor overrides."""
    return QueryAnalyzer(**overrides)


query_analyzer = QueryAnalyzer()


# -------------------------------- CLI (optional) -------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Query Analyzer CLI")
    parser.add_argument("--analyze", help="Analyze a single query")
    parser.add_argument("--search-strategy", help="Analyze and produce search strategy")
    parser.add_argument("--batch", help="Path to a file with one query per line")
    parser.add_argument("--output", help="Where to write batch JSON")

    args = parser.parse_args()

    async def _main():
        if args.analyze:
            a = await query_analyzer.analyze_query(args.analyze)
            print(json.dumps(a.to_dict(), indent=2))
            return

        if args.search_strategy:
            a = await query_analyzer.analyze_query(args.search_strategy)
            s = await query_analyzer.get_search_strategy(a.to_dict())
            print(json.dumps(s, indent=2))
            return

        if args.batch:
            out = []
            try:
                with open(args.batch, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                for q in lines:
                    try:
                        a = await query_analyzer.analyze_query(q)
                        out.append(a.to_dict())
                        print(f"Analyzed: {q[:60]}...")
                    except Exception as e:
                        print(f"Error analyzing: {q[:60]}... -> {e}")
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(out, f, indent=2)
                    print(f"Saved results to {args.output}")
                else:
                    print(json.dumps(out, indent=2))
            except FileNotFoundError:
                print(f"File not found: {args.batch}")
            return

        parser.print_help()

    asyncio.run(_main())
