# source/services/prompt_manager.py
# type: ignore
from __future__ import annotations

"""
Prompt Manager Service
----------------------

Centralized, config-driven prompt management for all LLM interactions.

Highlights
- System prompts (by key) and templated prompts (with required/optional fields)
- RAG utilities: build system+user prompts and compact context from sources
- Validation & optimization helpers (rough token estimates)
- Usage metrics & simple prompt cache
- Friendly to both sync/async callers (routers wrap with _maybe_await)
- Config-aware: pulls defaults and optional prompt sets from config.yaml

This module intentionally DOES NOT read environment variables.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ----------------------------- Config wiring ----------------------------- #

try:
    from source.config import (
        load_config,
        load_llm_settings,
    )
except Exception:  # pragma: no cover
    # Minimal fallbacks for isolated tests
    def load_config() -> Dict[str, Any]:
        return {}

    def load_llm_settings() -> Dict[str, Any]:
        return {}

_CFG = load_config() or {}
_LLM_SETTINGS = load_llm_settings() or {}

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ----------------------------- Data models ----------------------------- #

@dataclass
class PromptTemplate:
    """Template definition for dynamic prompt generation."""
    name: str
    template: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    category: str = "general"
    max_length: int = 4000
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate_fields(self, **kwargs) -> Tuple[bool, List[str]]:
        missing = [f for f in (self.required_fields or []) if f not in kwargs]
        return (len(missing) == 0), missing

    def estimate_tokens(self, **kwargs) -> int:
        try:
            rendered = self.template.format(**kwargs)
            return max(0, len(rendered) // 4)  # very rough English token heuristic
        except Exception:
            return max(0, len(self.template) // 4)


@dataclass
class PromptMetrics:
    """Simple usage metrics per template."""
    template_name: str
    usage_count: int = 0
    avg_response_time: float = 0.0
    avg_tokens_used: int = 0
    success_rate: float = 1.0
    last_used: Optional[str] = None

    def update(self, response_time: float, tokens_used: int, success: bool) -> None:
        self.usage_count += 1
        # running average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.usage_count - 1)) + float(response_time)
        ) / self.usage_count
        # running average tokens
        self.avg_tokens_used = int(
            ((self.avg_tokens_used * (self.usage_count - 1)) + int(tokens_used)) / self.usage_count
        )
        # running success rate
        s = 1.0 if success else 0.0
        self.success_rate = ((self.success_rate * (self.usage_count - 1)) + s) / self.usage_count
        self.last_used = datetime.now().isoformat()


# ----------------------------- Service ----------------------------- #

class PromptManagerService:
    """
    Centralized prompt manager (system prompts + templates), with:
    - RAG prompt builder utilities
    - Length validation/optimization
    - Metrics & cache
    """

    def __init__(self) -> None:
        self._system_prompts: Dict[str, str] = {}
        self._templates: Dict[str, PromptTemplate] = {}
        self._metrics: Dict[str, PromptMetrics] = {}
        self._cache: Dict[str, str] = {}
        self._cache_ttl_sec: int = 3600
        self._last_cache_clear: Optional[str] = None

        # Bootstrap defaults then merge config-provided prompts/templates
        self._load_defaults()
        self._load_from_config(_CFG)

        logger.info(
            "PromptManager ready: system_prompts=%d, templates=%d",
            len(self._system_prompts), len(self._templates),
        )

    # -------------------- Loaders -------------------- #

    def _load_defaults(self) -> None:
        # Baseline system prompts
        defaults = {
            "chat_with_rag": (
                "You are an expert AI assistant that answers questions using provided document "
                "context. Always prioritize the context; if it’s insufficient, say so explicitly. "
                "Cite sources using [Source N]. Be precise, helpful, and acknowledge uncertainty."
            ),
            "chat_without_rag": (
                "You are a helpful, knowledgeable, and concise AI assistant. Provide accurate, "
                "well-structured responses and state limitations when needed."
            ),
            "chat_streaming": (
                "You are optimized for real-time streaming. Use short, structured chunks and "
                "clear formatting for incremental delivery."
            ),
            "technical_expert": (
                "You are a technical expert. Provide accurate, detailed explanations with "
                "practical examples where helpful."
            ),
            "helpful_assistant": (
                "You are a helpful, accurate, and concise AI assistant. Provide useful, "
                "well-structured responses."
            ),
            "query_analysis": (
                "You are an expert query analysis system. Analyze user intent, complexity, and "
                "information requirements. Output ONLY valid JSON that matches the requested schema. "
                "Be thorough yet concise; keep confidence realistic."
            ),
        }
        self._system_prompts.update(defaults)

        # Baseline templates
        self._templates.update({
            "chat_with_context": PromptTemplate(
                name="chat_with_context",
                description="Standard RAG chat prompt with source citations",
                category="chat",
                version="1.0",
                required_fields=["context", "query"],
                optional_fields=[],
                max_length=4000,
                tags=["rag", "citations", "context-aware"],
                template=(
                    "Context Information:\n{context}\n\n"
                    "Question: {query}\n\n"
                    "Instructions: Answer using ONLY the context above. If the context is insufficient, say so explicitly.\n"
                    "Cite sources like [Source N] when referencing specific information.\n\n"
                    "Response:"
                ),
            ),
            "chat_with_history": PromptTemplate(
                name="chat_with_history",
                description="Chat with conversation history and context",
                category="chat",
                version="1.0",
                required_fields=["history", "context", "query"],
                optional_fields=["max_history_turns"],
                max_length=4000,
                tags=["history", "context", "conversational"],
                template=(
                    "Conversation History:\n{history}\n\n"
                    "Current Context:\n{context}\n\n"
                    "User Question: {query}\n\n"
                    "Instructions: Use both the conversation history and current context to provide a comprehensive answer.\n"
                    "Maintain conversation flow while incorporating new context information.\n\n"
                    "Response:"
                ),
            ),
            "query_analysis_comprehensive": PromptTemplate(
                name="query_analysis_comprehensive",
                description="Comprehensive query analysis with detailed JSON schema",
                category="analysis",
                version="1.0",
                required_fields=["query"],
                optional_fields=["context"],
                max_length=5000,
                tags=["json", "comprehensive", "structured"],
                template=(
                    "Analyze this user query and return ONLY valid JSON:\n\n"
                    "Query: \"{query}\"\n"
                    "Context: {context}\n\n"
                    "Required JSON format:\n"
                    "{\n"
                    "  \"intent\": \"factual|analytical|procedural|creative|research|clarification|conversational\",\n"
                    "  \"complexity\": \"simple|moderate|complex|research_intensive\",\n"
                    "  \"context_requirement\": \"none|minimal|moderate|extensive\",\n"
                    "  \"key_concepts\": [\"concept1\",\"concept2\"],\n"
                    "  \"sub_questions\": [\"sub1\",\"sub2\"],\n"
                    "  \"search_terms\": [\"t1\",\"t2\"],\n"
                    "  \"confidence\": 0.0,\n"
                    "  \"reasoning\": \"explain briefly\",\n"
                    "  \"estimated_tokens\": 0,\n"
                    "  \"requires_multi_step\": false,\n"
                    "  \"domain_hints\": [],\n"
                    "  \"query_type\": \"question|request|command\",\n"
                    "  \"ambiguity_score\": 0.0,\n"
                    "  \"suggested_approach\": \"rag|reasoning|direct|creative\",\n"
                    "  \"expected_answer_type\": \"factual|list|explanation|comparison|procedure\"\n"
                    "}\n"
                ),
            ),
            "reasoning_plan_detailed": PromptTemplate(
                name="reasoning_plan_detailed",
                description="Detailed reasoning plan",
                category="reasoning",
                version="1.0",
                required_fields=["query"],
                optional_fields=["complexity", "intent", "concepts", "sub_questions", "domain_hints"],
                max_length=5000,
                tags=["planning", "multi-step", "structured"],
                template=(
                    "Create a detailed reasoning plan for:\n"
                    "\"{query}\"\n\n"
                    "Return JSON with steps, dependencies, validation, fallback strategy, and confidence."
                ),
            ),
            "final_synthesis": PromptTemplate(
                name="final_synthesis",
                description="Final synthesis from reasoning steps",
                category="reasoning",
                version="1.0",
                required_fields=["original_query", "step_results"],
                optional_fields=["synthesis_instructions"],
                max_length=5000,
                tags=["synthesis", "integration", "final-answer"],
                template=(
                    "Synthesize a comprehensive final answer.\n\n"
                    "Original Query: {original_query}\n"
                    "Step Results:\n"
                    "{step_results}\n\n"
                    "Instructions: {synthesis_instructions}\n\n"
                    "Final Answer:"
                ),
            ),
            "document_summary_structured": PromptTemplate(
                name="document_summary_structured",
                description="Structured document summarization",
                category="document",
                version="1.0",
                required_fields=["document_id", "context"],
                optional_fields=["summary_type", "target_length", "include_key_points"],
                max_length=5000,
                tags=["summary", "structured"],
                template=(
                    "Create a structured summary of the document:\n\n"
                    "Document ID: {document_id}\n"
                    "Context:\n"
                    "{context}\n\n"
                    "Summary Requirements:\n"
                    "- Type: {summary_type}\n"
                    "- Target Length: {target_length}\n"
                    "- Include Key Points: {include_key_points}\n\n"
                    "Structure:\n"
                    "1) Main Topic/Purpose\n"
                    "2) Key Points\n"
                    "3) Supporting Details\n"
                    "4) Conclusions/Implications\n"
                    "5) Limitations\n\n"
                    "Summary:"
                ),
            ),
            "comparison_analysis": PromptTemplate(
                name="comparison_analysis",
                description="Structured comparison analysis",
                category="analysis",
                version="1.0",
                required_fields=["topic", "items", "context"],
                optional_fields=["criteria"],
                max_length=5000,
                tags=["comparison", "evaluation"],
                template=(
                    "Perform a comparison.\n\n"
                    "Topic: {topic}\n"
                    "Items: {items}\n"
                    "Context:\n"
                    "{context}\n"
                    "Criteria: {criteria}\n\n"
                    "Output:\n"
                    "1) Overview\n"
                    "2) Detailed Comparison\n"
                    "3) Strengths & Weaknesses\n"
                    "4) Summary Table (concise)\n"
                    "5) Recommendations"
                ),
            ),
        })

    def _load_from_config(self, cfg: Dict[str, Any]) -> None:
        """
        Merge prompts/templates from config if present.
        Supports the structure:

        prompts:
          system:
            key: "prompt text"
          templates:
            some_template:
              name: some_template
              template: "... {var} ..."
              required_fields: [...]
              optional_fields: [...]
              description: ...
              version: "1.0"
              category: "chat"
              max_length: 4000
              tags: [...]
        """
        try:
            prompts_section = cfg.get("prompts") or {}
            # System prompts
            sys_prompts = prompts_section.get("system") or {}
            if isinstance(sys_prompts, dict):
                for key, text in sys_prompts.items():
                    if isinstance(text, str) and text.strip():
                        self._system_prompts[key] = text

            # Templates
            templates = prompts_section.get("templates") or {}
            if isinstance(templates, dict):
                for key, tdict in templates.items():
                    if not isinstance(tdict, dict):
                        continue
                    try:
                        # Allow missing name key, default to map key
                        data = dict(tdict)
                        if "name" not in data:
                            data["name"] = key
                        tpl = PromptTemplate(
                            name=data.get("name", key),
                            template=data.get("template", ""),
                            required_fields=list(data.get("required_fields", [])),
                            optional_fields=list(data.get("optional_fields", [])),
                            description=data.get("description", ""),
                            version=str(data.get("version", "1.0")),
                            category=str(data.get("category", "general")),
                            max_length=int(data.get("max_length", 4000)),
                            tags=list(data.get("tags", [])),
                        )
                        self._templates[tpl.name] = tpl
                    except Exception as e:
                        logger.warning("Failed to load template '%s' from config: %s", key, e)
        except Exception as e:  # pragma: no cover
            logger.warning("Error merging prompts from config: %s", e)

    # -------------------- Health / defaults -------------------- #

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "system_prompts": len(self._system_prompts),
            "templates": len(self._templates),
        }

    def get_default_max_tokens(self) -> int:
        try:
            return int(_LLM_SETTINGS.get("max_tokens_default", 4000))
        except Exception:
            return 4000

    # -------------------- System prompts -------------------- #

    def get_system_prompt(self, prompt_type: str) -> str:
        return self._system_prompts.get(prompt_type, self._system_prompts.get("helpful_assistant", ""))

    def list_system_prompts(self) -> List[str]:
        return sorted(self._system_prompts.keys())

    def search_system_prompts(self, keyword: str) -> Dict[str, str]:
        kw = (keyword or "").lower().strip()
        out: Dict[str, str] = {}
        if not kw:
            return out
        for k, v in self._system_prompts.items():
            if kw in k.lower() or kw in v.lower():
                out[k] = v
        return out

    def add_system_prompt(self, prompt_type: str, prompt: str, *, overwrite: bool = False) -> bool:
        if not overwrite and prompt_type in self._system_prompts:
            return False
        self._system_prompts[prompt_type] = prompt
        return True

    # -------------------- Templates (CRUD) -------------------- #

    def get_template(self, name: str) -> Optional[Union[PromptTemplate, Dict[str, Any]]]:
        return self._templates.get(name)

    def list_templates(self, *, category: Optional[str] = None) -> List[str]:
        names = [n for n, t in self._templates.items() if (category is None or t.category == category)]
        return sorted(names)

    def get_templates_by_category(self) -> Dict[str, List[str]]:
        cat: Dict[str, List[str]] = {}
        for n, t in self._templates.items():
            cat.setdefault(t.category, []).append(n)
        for k in cat:
            cat[k].sort()
        return cat

    def add_template(self, template_like: Union[PromptTemplate, Dict[str, Any]], *, overwrite: bool = False) -> bool:
        """
        Accepts either a PromptTemplate or a dict with compatible keys.
        """
        if isinstance(template_like, PromptTemplate):
            name = template_like.name
            if not overwrite and name in self._templates:
                return False
            self._templates[name] = template_like
            return True

        # dict path
        data = dict(template_like or {})
        name = data.get("name")
        if not name:
            raise ValueError("Template 'name' is required")
        if not overwrite and name in self._templates:
            return False

        tpl = PromptTemplate(
            name=name,
            template=data.get("template", ""),
            required_fields=list(data.get("required_fields", [])),
            optional_fields=list(data.get("optional_fields", [])),
            description=data.get("description", ""),
            version=str(data.get("version", "1.0")),
            category=str(data.get("category", "general")),
            max_length=int(data.get("max_length", 4000)),
            tags=list(data.get("tags", [])),
        )
        self._templates[name] = tpl
        return True

    # -------------------- Prompt building -------------------- #

    def build_prompt(self, template_name: str, *, track_metrics: bool = True, **fields: Any) -> str:
        tpl = self._templates.get(template_name)
        if not tpl:
            raise ValueError(f"Template '{template_name}' not found")

        ok, missing = tpl.validate_fields(**fields)
        if not ok:
            raise ValueError(f"Missing required fields for template '{template_name}': {missing}")

        # deterministic cache key
        try:
            items = sorted(fields.items(), key=lambda x: x[0])
            cache_key = f"{template_name}:{hash(str(items))}"
        except Exception:
            cache_key = f"{template_name}:{time.time_ns()}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()
        success = True
        tokens = 0
        try:
            rendered = tpl.template.format(**fields)
            tokens = tpl.estimate_tokens(**fields)
            self._cache[cache_key] = rendered
            return rendered
        except KeyError as e:
            success = False
            raise ValueError(f"Template formatting error for '{template_name}': missing field {e!s}") from e
        except Exception as e:
            success = False
            raise ValueError(f"Error building template '{template_name}': {e}") from e
        finally:
            if track_metrics:
                self._update_metrics(template_name, response_time=time.time() - t0, tokens=tokens, success=success)

    def build_prompt_safe(self, template_name: str, **fields: Any) -> Tuple[bool, str]:
        try:
            return True, self.build_prompt(template_name, **fields)
        except Exception as e:
            logger.warning("build_prompt_safe failed for '%s': %s", template_name, e)
            return False, str(e)

    # -------------------- Specialized builders -------------------- #

    def build_rag_prompt(
        self,
        *,
        query: str,
        sources: List[Any],
        system_prompt_type: str = "chat_with_rag",
        template_name: str = "chat_with_context",
        max_context_length: int = 3000,
    ) -> Tuple[str, str]:
        context = self.build_context_from_sources(sources, max_length=max_context_length)
        system_prompt = self.get_system_prompt(system_prompt_type)
        user_prompt = self.build_prompt(template_name, query=query, context=context)
        return system_prompt, user_prompt

    def build_analysis_prompt(self, query: str, *, context: str = "") -> str:
        return self.build_prompt(
            "query_analysis_comprehensive",
            query=query,
            context=context or "No additional context provided.",
        )

    def build_planning_prompt(self, analysis: Union[Dict[str, Any], Any]) -> str:
        """
        Accepts the analysis object (as dict or object) and builds a planning prompt.
        """
        if isinstance(analysis, dict):
            query = analysis.get("original_query", "")
            complexity = (analysis.get("complexity") or "complex")
            intent = (analysis.get("intent") or "research")
            concepts = analysis.get("key_concepts", [])
            sub_questions = analysis.get("sub_questions", [])
            domain_hints = analysis.get("domain_hints", [])
        else:
            query = getattr(analysis, "original_query", "")
            # accept enum or string
            complexity_val = getattr(analysis, "complexity", "complex")
            intent_val = getattr(analysis, "intent", "research")
            complexity = getattr(complexity_val, "value", complexity_val)
            intent = getattr(intent_val, "value", intent_val)
            concepts = getattr(analysis, "key_concepts", [])
            sub_questions = getattr(analysis, "sub_questions", [])
            domain_hints = getattr(analysis, "domain_hints", [])

        return self.build_prompt(
            "reasoning_plan_detailed",
            query=query,
            complexity=complexity,
            intent=intent,
            concepts=concepts,
            sub_questions=sub_questions,
            domain_hints=domain_hints,
        )

    def build_synthesis_prompt(
        self,
        *,
        original_query: str,
        step_results: str,
        synthesis_instructions: str = "",
    ) -> str:
        return self.build_prompt(
            "final_synthesis",
            original_query=original_query,
            step_results=step_results,
            synthesis_instructions=(
                synthesis_instructions or "Combine findings into a coherent, comprehensive answer."
            ),
        )

    def build_document_summary_prompt(
        self,
        *,
        document_id: str,
        context: str,
        summary_type: str = "comprehensive",
        target_length: str = "medium",
        include_key_points: bool = True,
    ) -> str:
        return self.build_prompt(
            "document_summary_structured",
            document_id=document_id,
            context=context,
            summary_type=summary_type,
            target_length=target_length,
            include_key_points=str(include_key_points),
        )

    def build_comparison_prompt(
        self,
        *,
        topic: str,
        items: List[str],
        context: str,
        criteria: Optional[List[str]] = None,
    ) -> str:
        items_str = ", ".join(items) if isinstance(items, list) else str(items)
        criteria_str = ", ".join(criteria) if criteria else "relevance, effectiveness, cost, ease of use"
        return self.build_prompt(
            "comparison_analysis",
            topic=topic,
            items=items_str,
            context=context,
            criteria=criteria_str,
        )

    # -------------------- Context helpers (RAG) -------------------- #

    def build_context_from_sources(
        self,
        sources: List[Any],
        *,
        max_length: int = 3000,
        include_metadata: bool = True,
    ) -> str:
        """
        Compact context builder with per-source [Source N] headings.
        Accepts list of dicts or objects with fields: content/text, document_id, page_number, score/similarity_score, path, title, url.
        """
        if not sources:
            return "No relevant context information available."

        def _src_text(src: Any) -> str:
            if isinstance(src, dict):
                return str(src.get("content") or src.get("text") or "")
            if hasattr(src, "content"):
                return str(getattr(src, "content"))
            if hasattr(src, "text"):
                return str(getattr(src, "text"))
            return str(src)

        def _src_meta(src: Any) -> Dict[str, Any]:
            keys = ("document_id", "page_number", "score", "similarity_score", "path", "title", "url")
            out: Dict[str, Any] = {}
            if isinstance(src, dict):
                for k in keys:
                    if k in src and src[k] is not None:
                        out[k] = src[k]
                return out
            for k in keys:
                if hasattr(src, k):
                    val = getattr(src, k)
                    if val is not None:
                        out[k] = val
            return out

        parts: List[str] = []
        used = 0
        for i, s in enumerate(sources, 1):
            text = _src_text(s).strip()
            if not text:
                continue
            meta = _src_meta(s) if include_metadata else {}
            head_bits: List[str] = []
            if "document_id" in meta:
                head_bits.append(f"Document: {meta['document_id']}")
            if "page_number" in meta:
                head_bits.append(f"Page {meta['page_number']}")
            if "score" in meta or "similarity_score" in meta:
                sc = meta.get("score", meta.get("similarity_score"))
                try:
                    head_bits.append(f"Score: {float(sc):.3f}")
                except Exception:
                    pass
            if "path" in meta:
                head_bits.append(f"Path: {meta['path']}")
            if "title" in meta:
                head_bits.append(f"Title: {meta['title']}")
            if "url" in meta:
                head_bits.append(f"URL: {meta['url']}")

            head = f"[Source {i}]"
            if head_bits:
                head += " (" + ", ".join(head_bits) + ")"
            entry = f"{head}:\n{text}\n"
            if used + len(entry) > max_length and parts:
                break
            parts.append(entry)
            used += len(entry)

        if not parts:
            return "No valid source content available."

        context = "\n".join(parts)
        if len(parts) < len(sources):
            context += f"\n\n[Note: Showing {len(parts)} of {len(sources)} sources due to length constraints]"
        return context

    # -------------------- Validation / optimization -------------------- #

    def validate_prompt_length(self, prompt: str, *, max_tokens: int = 4000) -> Dict[str, Any]:
        est = max(0, len(prompt) // 4)
        chars = len(prompt)
        lines = prompt.count("\n") + 1
        is_valid = est <= max_tokens
        ratio = (est / max_tokens) if max_tokens > 0 else 1.0
        severity = "ok"
        suggestions: List[str] = []
        if not is_valid:
            severity = "error"
            suggestions.extend([
                f"Prompt exceeds token limit by {est - max_tokens} tokens.",
                "Shorten context or split into multiple turns.",
                "Summarize long sections before inclusion.",
            ])
        elif ratio > 0.8:
            severity = "warning"
            suggestions.append("Approaching token limit; consider condensing.")

        if chars > 10000:
            suggestions.append("Very long prompt; consider breaking into sections.")
        if lines > 100:
            suggestions.append("Many line breaks; check formatting for efficiency.")

        return {
            "character_count": chars,
            "estimated_tokens": est,
            "line_count": lines,
            "max_tokens": max_tokens,
            "is_valid": is_valid,
            "token_usage_ratio": ratio,
            "severity": severity,
            "suggestions": suggestions,
        }

    def optimize_prompt_length(self, prompt: str, *, target_tokens: int = 3000) -> str:
        current = max(0, len(prompt) // 4)
        if current <= target_tokens:
            return prompt

        # Simple middle summarization strategy
        lines = prompt.splitlines()
        if len(lines) <= 10:
            return (prompt[: target_tokens * 4]) + "...[truncated]"

        keep_start = max(1, int(len(lines) * 0.3))
        keep_end = max(1, int(len(lines) * 0.3))
        middle_omitted = len(lines) - (keep_start + keep_end)
        start_part = "\n".join(lines[:keep_start])
        end_part = "\n".join(lines[-keep_end:])
        middle_note = f"[Content summarized: {middle_omitted} lines omitted]"
        return f"{start_part}\n\n{middle_note}\n\n{end_part}"

    # -------------------- Metrics / cache -------------------- #

    def _update_metrics(self, template_name: str, *, response_time: float, tokens: int, success: bool) -> None:
        m = self._metrics.get(template_name)
        if not m:
            m = PromptMetrics(template_name=template_name)
            self._metrics[template_name] = m
        m.update(response_time=response_time, tokens_used=tokens, success=success)

    def get_template_metrics(self, template_name: str) -> Optional[PromptMetrics]:
        return self._metrics.get(template_name)

    def get_all_metrics(self) -> Dict[str, PromptMetrics]:
        return dict(self._metrics)

    def reset_metrics(self, template_name: Optional[str] = None) -> None:
        if template_name:
            self._metrics.pop(template_name, None)
        else:
            self._metrics.clear()

    def clear_cache(self) -> int:
        n = len(self._cache)
        self._cache.clear()
        self._last_cache_clear = datetime.now().isoformat()
        return n

    def get_cache_stats(self) -> Dict[str, Any]:
        size_bytes = 0
        try:
            size_bytes = sum((len(k) + len(v)) for k, v in self._cache.items())
        except Exception:
            pass
        return {
            "entries": len(self._cache),
            "size_bytes": size_bytes,
            "ttl_seconds": self._cache_ttl_sec,
            "last_cleared": self._last_cache_clear,
        }


# ----------------------------- Factory & Singleton ----------------------------- #

def get_prompt_manager() -> PromptManagerService:
    """Factory to allow dependency injection / testing."""
    return PromptManagerService()


# Global singleton used by most callers
prompt_manager = PromptManagerService()


# ----------------------------- CLI (optional) ----------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Prompt Manager CLI")
    p.add_argument("--list-prompts", action="store_true")
    p.add_argument("--list-templates", action="store_true")
    p.add_argument("--by-category", action="store_true")
    p.add_argument("--search", type=str)
    p.add_argument("--build", type=str, help="Template name to build")
    p.add_argument("--fields", type=str, help="JSON for fields when using --build", default="{}")
    p.add_argument("--validate", type=str, help="Validate a raw prompt string")
    args = p.parse_args()

    if args.list_prompts:
        for k in prompt_manager.list_system_prompts():
            print("-", k)
    if args.list_templates:
        for k in prompt_manager.list_templates():
            t = prompt_manager.get_template(k)
            print(f"- {k} ({t.category})")  # type: ignore
    if args.by_category:
        print(json.dumps(prompt_manager.get_templates_by_category(), indent=2))
    if args.search:
        print(json.dumps(prompt_manager.search_system_prompts(args.search), indent=2))
    if args.build:
        try:
            fields = json.loads(args.fields)
        except Exception:
            fields = {}
        ok, result = prompt_manager.build_prompt_safe(args.build, **fields)
        print("OK:" if ok else "ERR:", result if ok else result[:300])
    if args.validate:
        print(json.dumps(prompt_manager.validate_prompt_length(args.validate), indent=2))
