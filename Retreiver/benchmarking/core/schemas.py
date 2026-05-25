from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    id: int
    pdf_name: str
    paragraph: str
    parent_id: str = ""
    metadata: Dict[str, Any] | None = None


@dataclass
class QueryCase:
    id: int
    query: str
    expected_text: str
    category: str = "uncategorized"
    metadata: Dict[str, Any] | None = None


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class StageTiming:
    name: str
    seconds: float


def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
