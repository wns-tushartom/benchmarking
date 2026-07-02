# type: ignore
"""
Document parsing service with robust MinerU (magic-pdf) handling and PyPDF2 fallback.
- Anchors paths to the Retreiver project root (QA_Text/Retreiver)
- Fixes the nested output directory issue from MinerU.
- Adds richer diagnostics and configuration knobs.
- Works with bytes uploads for PDF/TXT/MD.

Usage (module-level singleton):
    from app.services.document_parser import document_parser

CLI bulk runner:
    python -m app.services.document_parser
"""
import os
import sys
import uuid
import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from source.config import load_document_parser
resource_param: Dict[str, Any] = load_document_parser() or {}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Path anchoring (anchor to Retreiver root) ----------------
# This file is: .../QA_Text/Retreiver/source/services/document_parser.py
# Retreiver root is parents[2]
APP_ROOT = Path(__file__).resolve().parents[2]              # .../QA_Text/Retreiver
DATA_ROOT = APP_ROOT / "data"

def _resolve_path(p: Optional[str | Path], fallback: Path) -> Path:
    """
    Resolve `p` relative to APP_ROOT when not absolute; use fallback if p is None/empty.
    Returns absolute Path.
    """
    if not p:
        path = fallback
    else:
        path = Path(p)
    return path if path.is_absolute() else (APP_ROOT / path).resolve()

# Config-driven paths (with sensible defaults)
OUTPUT_DIR = _resolve_path(resource_param.get("output_dir", "data/processed"), DATA_ROOT / "processed")
TMP_DIR    = _resolve_path(resource_param.get("tmp_dir", "data/tmp"), DATA_ROOT / "tmp")
INPUT_DIR  = _resolve_path(resource_param.get("input_dir", "data/pdfs"), DATA_ROOT / "pdfs")
OUT_PARSED = _resolve_path(resource_param.get("output_dir_parsed", "data/parsed_output"), DATA_ROOT / "parsed_output")

# Make sure key dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- MinerU import ----------------
try:
    from magic_pdf import pdf_parse_main  # type: ignore
except Exception:
    pdf_parse_main = None  # type: ignore

_cli_candidates = [
    shutil.which("magic-pdf"),
    shutil.which("mineru"),
    str(Path(sys.executable).resolve().parent / "magic-pdf"),
    str(Path(sys.executable).resolve().parent / "mineru"),
]
MAGIC_PDF_CLI = next((c for c in _cli_candidates if c and Path(c).exists()), None)
MINERU_AVAILABLE = bool(pdf_parse_main is not None or MAGIC_PDF_CLI)

# ---------------- PyPDF2 fallback ----------------
try:
    import PyPDF2  # type: ignore
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# ---------------- Supported types ----------------
# Merge YAML-provided list (cloud_services.supported_file_types) with sane defaults
_yaml_exts = resource_param.get("supported_exts") or resource_param.get("supported_file_types") or [".pdf", ".txt", ".md"]
SUPPORTED_EXTS = sorted(set([e.lower() for e in _yaml_exts]))

# ---------------- Other knobs ----------------
MAX_FILE_SIZE = int(resource_param.get("max_file_size", 50 * 1024 * 1024))
KEEP_TMP_DEFAULT = bool(resource_param.get("keep_tmp", False))
FORCE_BACKEND_DEFAULT = resource_param.get("force_backend")  # "mineru" | "pypdf2" | None
MINERU_PARSE_METHOD = resource_param.get("mineru_parse_method", "auto")
MINERU_MODEL_JSON_PATH = resource_param.get("mineru_model_json_path", None)

# ---------------- Schema ----------------
class DocumentType(Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"

@dataclass
class DocumentContent:
    page_number: int
    content: str
    images: List[str]
    tables: List[Dict[str, Any]]
    formulas: List[str]
    metadata: Dict[str, Any]

@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    document_type: DocumentType
    total_pages: int
    content: List[DocumentContent]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# ---------------- Service ----------------
class DocumentParserService:
    def __init__(
        self,
        output_dir: str | Path = OUTPUT_DIR,
        max_file_size: int = MAX_FILE_SIZE,
        keep_tmp_files: bool = KEEP_TMP_DEFAULT,
        force_backend: Optional[str] = FORCE_BACKEND_DEFAULT,  # one of {None, "mineru", "pypdf2"}
    ):
        # Resolve output_dir relative to APP_ROOT, if needed
        self.output_dir = _resolve_path(str(output_dir), OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._max_file_size = int(max_file_size)
        self._keep_tmp = bool(keep_tmp_files)
        self._force_backend = force_backend

        logger.info(
            "DocumentParserService initialized: out=%s, max_file_size=%d, force_backend=%s, keep_tmp=%s",
            self.output_dir,
            self._max_file_size,
            self._force_backend,
            self._keep_tmp,
        )
        logger.info("Backends — MinerU: %s, PyPDF2: %s", MINERU_AVAILABLE, PYPDF2_AVAILABLE)

    # ---------- Health/info ----------
    async def get_supported_formats(self) -> List[str]:
        """Used by /health endpoint."""
        return list(SUPPORTED_EXTS)

    async def get_max_file_size(self) -> int:
        return self._max_file_size

    # ---------- Public APIs ----------
    async def parse_pdf(self, file_path: str, filename: str, allow_fallback: bool = False) -> ParsedDocument:
        """Parse PDF using MinerU (preferred) or PyPDF2 when explicitly allowed."""
        document_id = str(uuid.uuid4())
        # Each parse session gets a UUID container directory
        session_dir = self.output_dir / document_id
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting PDF parsing for %s (DocID=%s)", filename, document_id)

        # Backend selection rules
        backend = self._decide_backend()
        logger.info("Selected backend: %s", backend)

        if backend == "mineru":
            try:
                return await self._parse_pdf_with_mineru(file_path, filename, document_id, session_dir)
            except Exception as e:
                if not allow_fallback:
                    raise RuntimeError(f"MinerU parsing failed and PyPDF2 fallback disabled: {e}") from e
                logger.warning("MinerU parsing failed, falling back to PyPDF2 because fallback is enabled: %s", e)

        if backend == "pypdf2" and not allow_fallback:
            raise RuntimeError("PyPDF2 fallback disabled. MinerU/magic-pdf is required for layout-aware extraction.")

        if PYPDF2_AVAILABLE and allow_fallback:
            return await self._parse_pdf_with_pypdf2(file_path, filename, document_id)

        # If we reach here, nothing could parse
        raise RuntimeError("No layout-aware PDF parser available. Install magic-pdf/MinerU or explicitly enable PyPDF2 text-only fallback.")

    async def parse_bytes(self, content: bytes, filename: str, mime_type: Optional[str] = None) -> ParsedDocument:
        """
        Generic entrypoint for uploaded files:
        - If PDF: write a temp file and call parse_pdf.
        - If .txt or .md: decode to text and wrap into a single-page ParsedDocument.
        """
        ext = Path(filename).suffix.lower()

        # Size check from config
        if len(content) > self._max_file_size:
            raise ValueError(f"File too large: {len(content)} bytes (max {self._max_file_size})")

        if ext == ".pdf":
            tmp_dir = TMP_DIR  # anchored tmp dir
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"{uuid.uuid4().hex}.pdf"
            tmp_path.write_bytes(content)
            try:
                return await self.parse_pdf(str(tmp_path), filename)
            finally:
                if not self._keep_tmp:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if ext in (".txt", ".md"):
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            doc_type = DocumentType.MARKDOWN if ext == ".md" else DocumentType.TEXT
            now = datetime.now()
            return ParsedDocument(
                document_id=str(uuid.uuid4()),
                filename=filename,
                document_type=doc_type,
                total_pages=1,
                content=[DocumentContent(
                    page_number=0,
                    content=text.strip(),
                    images=[], tables=[], formulas=[],
                    metadata={"extraction_method": "bytes_decode", "text_length": len(text)}
                )],
                metadata={"mime_type": mime_type, "parsing_method": "decode", "file_size": len(content)},
                created_at=now, updated_at=now
            )

        raise ValueError(f"Unsupported file type for parse_bytes: {ext}")

    async def validate_file(self, file_path: str, max_size: Optional[int] = None) -> bool:
        """Validate existence, size and type."""
        if not os.path.exists(file_path):
            raise ValueError("File does not exist")

        size = os.path.getsize(file_path)
        limit = int(max_size if max_size is not None else self._max_file_size)
        if size > limit:
            raise ValueError(f"File too large: {size} bytes (max {limit})")

        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file format: {ext}")

        return True

    async def validate_bytes(self, content: bytes, filename: str, max_size: Optional[int] = None) -> bool:
        """Validate uploaded file bytes."""
        size = len(content)
        limit = int(max_size if max_size is not None else self._max_file_size)
        if size > limit:
            raise ValueError(f"File too large: {size} bytes (max {limit})")

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file format: {ext}")

        return True

    # ---------- Internal helpers ----------
    def _decide_backend(self) -> str:
        if self._force_backend in {"mineru", "pypdf2"}:
            return self._force_backend
        # Default preference
        if MINERU_AVAILABLE:
            return "mineru"
        if PYPDF2_AVAILABLE:
            return "pypdf2"
        return "none"

    async def _parse_pdf_with_mineru(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        session_dir: Path,
    ) -> ParsedDocument:
        """Parse PDF using MinerU. Handles nested output directory and collects artifacts."""
        # Run MinerU in a worker thread (blocking)
        await asyncio.to_thread(self._run_mineru_parsing, file_path, str(session_dir))

        # MinerU output layout varies across versions. Prefer stem-matching
        # content_list files, but recursively discover any produced JSON.
        pdf_stem = Path(filename).stem
        produced = [p for p in session_dir.rglob("*")]
        content_candidates = sorted(
            [p for p in produced if p.is_file() and p.name.endswith("_content_list.json")],
            key=lambda p: (0 if pdf_stem.lower() in str(p).lower() else 1, len(str(p))),
        )
        if not content_candidates:
            try:
                produced_rel = [str(p.relative_to(session_dir)) for p in produced]
                logger.warning(
                    "MinerU content JSON not found. Session produced %d paths under %s.\nPaths:\n%s",
                    len(produced_rel),
                    session_dir,
                    "\n".join(produced_rel[:100]),
                )
            except Exception:
                pass
            raise RuntimeError("MinerU did not produce a *_content_list.json file. Layout-aware extraction is not usable.")

        content_file = content_candidates[0]
        mineru_dir = content_file.parent
        markdown_candidates = sorted(
            [p for p in produced if p.is_file() and p.suffix.lower() == ".md"],
            key=lambda p: (0 if pdf_stem.lower() in str(p).lower() else 1, len(str(p))),
        )
        markdown_file = markdown_candidates[0] if markdown_candidates else mineru_dir / f"{pdf_stem}.md"

        content_data: List[DocumentContent] = []
        try:
            with content_file.open("r", encoding="utf-8") as f:
                json_content = json.load(f)
            if isinstance(json_content, dict):
                json_content = json_content.get("content_list") or json_content.get("items") or []
            content_data = self._process_json_content(json_content)
        except Exception as e:
            logger.error("Failed reading MinerU JSON %s: %s", content_file, e)
            raise
        if not content_data:
            raise RuntimeError(f"MinerU produced no structured content from {content_file}. Refusing empty layout extraction.")

        markdown_content = ""
        if markdown_file.exists():
            try:
                with markdown_file.open("r", encoding="utf-8") as f:
                    markdown_content = f.read()
            except Exception as e:
                logger.error("Failed reading MinerU Markdown: %s", e)
        else:
            logger.info("MinerU Markdown not found (optional): %s", markdown_file)

        # Helpful diagnostics: list what MinerU actually produced
        try:
            produced = [str(p.relative_to(session_dir)) for p in mineru_dir.rglob("*")]
            logger.info(
                "MinerU produced %d files/dirs under %s:\n%s",
                len(produced),
                mineru_dir,
                "\n".join(produced[:200]),
            )
        except Exception:
            pass

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        now = datetime.now()
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            document_type=DocumentType.PDF,
            total_pages=len(content_data),
            content=content_data,
            metadata={
                "original_file_path": file_path,
                "session_dir": str(session_dir),
                "output_dir": str(mineru_dir),  # real output dir
                "file_size": file_size,
                "markdown_content": markdown_content,
                "parsing_method": "MinerU",
            },
            created_at=now,
            updated_at=now,
        )

    async def _parse_pdf_with_pypdf2(self, file_path: str, filename: str, document_id: str) -> ParsedDocument:
        """Parse PDF using PyPDF2."""
        content_data: List[DocumentContent] = []
        total_pages = 0

        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        content_data.append(
                            DocumentContent(
                                page_number=page_num,
                                content=text,
                                images=[],
                                tables=[],
                                formulas=[],
                                metadata={"text_length": len(text), "extraction_method": "PyPDF2"},
                            )
                        )
                except Exception as e:
                    logger.warning("PyPDF2: failed to extract page %d: %s", page_num, e)

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        now = datetime.now()
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            document_type=DocumentType.PDF,
            total_pages=len(content_data),
            content=content_data,
            metadata={
                "original_file_path": file_path,
                "file_size": file_size,
                "parsing_method": "PyPDF2_fallback",
                "text_only_review": True,
                "needs_ocr_review": True,
                "layout_counts_valid": False,
                "total_pdf_pages": total_pages,
            },
            created_at=now,
            updated_at=now,
        )

    def _run_mineru_parsing(self, file_path: str, output_path: str) -> None:
        """Run MinerU synchronously (called inside a thread)."""
        if not MINERU_AVAILABLE:
            raise RuntimeError("magic-pdf (MinerU) is not available")
        try:
            if pdf_parse_main is not None:
                # Older MinerU API path.
                pdf_parse_main(
                    pdf_path=file_path,
                    parse_method=MINERU_PARSE_METHOD,
                    model_json_path=MINERU_MODEL_JSON_PATH,
                    is_json_md_dump=True,
                    output_dir=output_path,
                )
                logger.info("MinerU parsing completed for %s", file_path)
                return

            if not MAGIC_PDF_CLI:
                raise RuntimeError("magic-pdf CLI is not available")
            cmd = [
                MAGIC_PDF_CLI,
                "-p", file_path,
                "-o", output_path,
                "-m", str(MINERU_PARSE_METHOD or "auto"),
            ]
            timeout_seconds = int(resource_param.get("mineru_timeout_seconds", 3600))
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            result = subprocess.run(
                cmd,
                cwd=str(APP_ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            if result.returncode != 0 or any(marker in result.stderr for marker in ("Traceback", "FileNotFoundError", "DownloadModelError")):
                logger.error("MinerU CLI stdout:\n%s", result.stdout[-4000:])
                logger.error("MinerU CLI stderr:\n%s", result.stderr[-4000:])
                raise RuntimeError(f"MinerU CLI failed with exit code {result.returncode}")
            if result.stdout:
                logger.info("MinerU CLI stdout:\n%s", result.stdout[-2000:])
            if result.stderr:
                logger.info("MinerU CLI stderr:\n%s", result.stderr[-2000:])
            logger.info("MinerU CLI parsing completed for %s", file_path)
        except Exception as e:
            logger.error("MinerU parsing failed: %s", e)
            raise

    def _process_json_content(self, json_content: List[Dict[str, Any]]) -> List[DocumentContent]:
        """Convert MinerU JSON output into structured content."""
        content_by_page: Dict[int, Dict[str, List[Any]]] = defaultdict(
            lambda: {"text": [], "images": [], "tables": [], "formulas": []}
        )

        for item in json_content:
            page_num = item.get("page_idx", 0)
            try:
                page_num = int(page_num)
            except Exception:
                page_num = 0

        #   Type routing
            typ = str(item.get("type") or item.get("category") or item.get("block_type") or "text").lower()
            if typ in {"text", "title", "plain_text", "list", "list_item"} or "text" in typ:
                txt = str(item.get("text") or item.get("content") or item.get("page_content") or "").strip()
                if txt:
                    content_by_page[page_num]["text"].append(txt)
            elif typ in {"image", "img", "figure"} or "image" in typ or typ == "figure":
                p = str(item.get("image_path") or item.get("img_path") or item.get("path") or "").strip()
                if p:
                    content_by_page[page_num]["images"].append(p)
                caption = str(item.get("caption") or item.get("image_caption") or "").strip()
                if caption:
                    content_by_page[page_num]["text"].append(caption)
            elif "table" in typ:
                content_by_page[page_num]["tables"].append(item)
                table_text = str(item.get("table_body") or item.get("table_caption") or item.get("text") or item.get("content") or "").strip()
                if table_text:
                    content_by_page[page_num]["text"].append(table_text)
            elif "formula" in typ or "equation" in typ:
                latex = str(item.get("latex") or item.get("text") or item.get("content") or "").strip()
                if latex:
                    content_by_page[page_num]["formulas"].append(latex)

        content_list: List[DocumentContent] = []
        for page_num in sorted(content_by_page.keys()):
            d = content_by_page[page_num]
            text_content = " ".join(d["text"]).strip()
            content_list.append(
                DocumentContent(
                    page_number=page_num,
                    content=text_content,
                    images=d["images"],
                    tables=d["tables"],
                    formulas=d["formulas"],
                    metadata={
                        "text_length": len(text_content),
                        "image_count": len(d["images"]),
                        "table_count": len(d["tables"]),
                        "formula_count": len(d["formulas"]),
                    },
                )
            )
        return content_list

# ---------------- Module-level instance ----------------
# This is what your routers import:  from app.services.document_parser import document_parser
document_parser = DocumentParserService()

# ---------------- Bulk runner (optional) ----------------
async def process_pdf_folder(
    input_dir: str | Path = INPUT_DIR,
    output_dir: str | Path = OUT_PARSED
):
    # Resolve both relative to APP_ROOT if needed
    input_dir_path = _resolve_path(str(input_dir), INPUT_DIR)
    output_dir_path = _resolve_path(str(output_dir), OUT_PARSED)

    os.makedirs(input_dir_path, exist_ok=True)
    os.makedirs(output_dir_path, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in", input_dir_path)
        return

    # Reuse the module singleton so config is consistent
    parser = document_parser

    for pdf_file in pdf_files:
        file_path = os.path.join(input_dir_path, pdf_file)
        try:
            await parser.validate_file(file_path)
            parsed_doc = await parser.parse_pdf(file_path, pdf_file)

            # Save Markdown
            md_output_path = os.path.join(output_dir_path, f"{Path(pdf_file).stem}.md")
            with open(md_output_path, "w", encoding="utf-8") as f:
                if parsed_doc.metadata.get("markdown_content"):
                    f.write(parsed_doc.metadata["markdown_content"])
                else:
                    for page in parsed_doc.content:
                        f.write(f"# Page {page.page_number + 1}\n\n{page.content}\n\n")

            # Save JSON (structured metadata)
            json_output_path = os.path.join(output_dir_path, f"{Path(pdf_file).stem}.json")
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_doc.metadata, f, indent=2)

            print(f"Processed {pdf_file} -> {md_output_path} + {json_output_path}")
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")

if __name__ == "__main__":
    async def main():
        print("Bulk PDF Processing")
        print("=" * 50)
        print(f"Processing PDFs from: {INPUT_DIR}")
        print(f"Saving output to:    {OUT_PARSED}")
        await process_pdf_folder(INPUT_DIR, OUT_PARSED)
    asyncio.run(main())
