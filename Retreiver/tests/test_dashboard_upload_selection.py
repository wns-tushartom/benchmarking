from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess

import pytest

from scripts import serve_benchmark_dashboard as dashboard
from source.services import project_documents
from source.services.project_documents import (
    ProjectDocumentStorageError,
    extract_project_documents,
)


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.js"


def _browser_matrix_request(project_id: str, groundtruth_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": f"project:{project_id}",
        "groundtruth_id": groundtruth_id,
        "typed_queries": [],
        "top_k": 3,
        "selections": {
            "chunkers": ["fixed_tok1200_ov150"],
            "embeddings": ["jina_v3"],
            "vector_stores": ["FAISS"],
            "rerankers": [],
        },
        "large_matrix_confirmation": None,
    }


class _JsonRequest:
    def __init__(self, path: str, payload: object):
        body = json.dumps(payload).encode("utf-8")
        self.path = path
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        self.rfile = BytesIO(body)
        self.responses: list[tuple[int, dict[str, object]]] = []
        self.close_connection = False

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        self.responses.append((status, payload))


def test_mismatched_dataset_groundtruth_returns_structured_400_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    projects = root / "data" / "user_projects"
    projects.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "ROOT", root)
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", projects)
    monkeypatch.setattr(dashboard, "JOB_DIR", root / "data" / "dashboard_jobs")
    project = dashboard.create_user_project_upload(
        "policy.txt", b"Refunds are available for thirty days.", "Customer policy"
    )
    official = root / "data" / "groundtruth" / "qa_text_test.csv"
    official.parent.mkdir(parents=True)
    official.write_text(
        "query,answer,source\nWhat is the refund rule?,Thirty days,policy.txt\n",
        encoding="utf-8",
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda command: launched.append(command) or {"job_id": "must-not-launch"},
    )
    request = _JsonRequest(
        "/api/run/project-matrix",
        _browser_matrix_request(
            str(project["project_id"]),
            "groundtruth:repository:qa_text_test.csv",
        ),
    )

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert launched == []
    assert len(request.responses) == 1
    status, payload = request.responses[0]
    assert status == 400
    assert payload["error"]["code"] == "invalid_project_matrix_request"  # type: ignore[index]
    assert payload["error"]["request_id"]  # type: ignore[index]
    assert str(root) not in json.dumps(payload)


def test_frontend_auto_selects_and_locks_dataset_owned_groundtruth_across_mirrors() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  let html = '';
  const element = {{
    id, value: '', textContent: '', disabled: false, hidden: false, checked: false,
    options: [], selectedOptions: [],
    classList: {{toggle() {{}}, add() {{}}, remove() {{}}}},
    addEventListener() {{}}, querySelectorAll() {{return [];}}, setAttribute() {{}}, replaceChildren() {{}},
  }};
  Object.defineProperty(element, 'innerHTML', {{
    get() {{ return html; }},
    set(value) {{
      html = String(value);
      if (!html.includes('<option')) return;
      element.options = [...html.matchAll(/<option value="([^"]+)"([^>]*)>([^<]*)<\\/option>/g)].map(match => ({{
        value: match[1], disabled: match[2].includes('disabled'), textContent: match[3],
      }}));
      element.selectedOptions = element.options.filter(option => option.value === element.value);
    }},
  }});
  return element;
}}
const document = {{
  getElementById(id) {{ return elements[id] || (elements[id] = makeElement(id)); }},
  querySelectorAll() {{ return []; }}, addEventListener() {{}},
  createElement(tag) {{ return makeElement(tag); }},
  body: {{insertAdjacentHTML() {{}}}},
}};
const context = {{
  console, document, window: {{}}, location: {{hash:'#metrics'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok:true,json:async()=>({{}}),text:async()=>''}}),
  setTimeout() {{}}, AbortController, URLSearchParams,
}};
vm.createContext(context);
const app = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(app + `
state = {{operational: {{}}, files: [], options: {{}}, sourceCatalog: {{
  datasets: [
    {{id:'dataset:wns-default',label:'Official WNS',kind:'default',ready:true,document_count:2,linked_groundtruth_id:'groundtruth:repository:qa_text_test.csv'}},
    {{id:'project:alpha',label:'Alpha corpus',kind:'uploaded_project',ready:true,document_count:1,linked_groundtruth_id:'groundtruth:project:alpha'}},
    {{id:'project:beta',label:'Beta corpus',kind:'uploaded_project',ready:true,document_count:1,linked_groundtruth_id:'groundtruth:none',mode:'evidence_only'}},
  ],
  groundtruth: [
    {{id:'groundtruth:none',label:'None — evidence-only',valid:true,row_count:0}},
    {{id:'groundtruth:repository:qa_text_test.csv',label:'Official benchmark GT',valid:true,row_count:500}},
    {{id:'groundtruth:project:alpha',label:'Alpha GT',valid:true,row_count:3}},
  ],
}}}};
renderSourceSelectors(state.sourceCatalog);
document.getElementById('globalDataset').value = 'project:alpha';
const linked = syncLinkedGroundtruth();
applyGlobalSourceContext({{syncResults:false}});
const snapshot = id => ({{
  value: document.getElementById(id).value,
  disabled: document.getElementById(id).disabled,
  options: document.getElementById(id).options.map(option => ({{value:option.value,disabled:option.disabled,text:option.textContent}})),
}});
globalThis.__result = {{
  linked,
  global: snapshot('globalGroundtruth'),
  run: snapshot('runGroundtruth'),
  nvidia: snapshot('nvidiaGroundtruth'),
  globalContext: document.getElementById('globalSourceContext').textContent,
  runContext: document.getElementById('runSourceContext').textContent,
}};
`, context);
const result = context.__result;
if (result.linked !== 'groundtruth:project:alpha') throw new Error(JSON.stringify(result));
for (const mirror of [result.global, result.run, result.nvidia]) {{
  if (mirror.value !== result.linked || !mirror.disabled) throw new Error(JSON.stringify(result));
  const linked = mirror.options.find(option => option.value === result.linked);
  if (!linked || linked.disabled || !linked.text.includes('Alpha GT')) throw new Error(JSON.stringify(result));
  if (mirror.options.some(option => option.value !== result.linked && !option.disabled)) throw new Error(JSON.stringify(result));
}}
if (!result.globalContext.includes('Alpha corpus') || !result.globalContext.includes('Alpha GT')) throw new Error(JSON.stringify(result));
if (!result.runContext.includes('Alpha corpus') || !result.runContext.includes('Alpha GT')) throw new Error(JSON.stringify(result));
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_evidence_only_metrics_suppress_quality_accuracy_and_winner_claims() -> None:
    app = APP.read_text(encoding="utf-8")
    assert "payload?.scoring_mode !== 'retrieval_labels'" in app
    assert "Recall, MRR, nDCG, score, and winner claims are intentionally suppressed" in app
    assert "renderBestMethods([])" in app


def test_mineru_pdf_documents_preserve_page_table_image_text_and_parser_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    pdf = project / "raw_uploads" / "policy.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF fake input")

    class Page:
        page_number = 0
        content = "Layout paragraph. Image-derived refund diagram."
        images = ["images/refund-diagram.png"]
        tables = [{"table_body": "| Rule | Days |\n|---|---|\n| Refund | 30 |"}]

    class Parsed:
        content = [Page()]
        metadata = {"parsing_method": "MinerU"}

    monkeypatch.setattr(
        project_documents, "_parse_pdf_with_mineru", lambda *_args, **_kwargs: Parsed(), raising=False
    )

    documents = extract_project_documents(project, [pdf])

    assert len(documents) == 1
    assert documents[0].metadata == {"page_number": "1", "parser_method": "MinerU"}
    assert "Image-derived refund diagram" in documents[0].text
    assert "| Refund | 30 |" in documents[0].text
    assert "MinerU image regions: 1" in documents[0].text


def test_mineru_failure_is_visible_and_never_silently_uses_text_only_pdf_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    pdf = project / "raw_uploads" / "policy.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF fake input")
    monkeypatch.setattr(
        project_documents,
        "_parse_pdf_with_mineru",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProjectDocumentStorageError("MinerU unavailable")
        ),
        raising=False,
    )

    with pytest.raises(ProjectDocumentStorageError, match="MinerU"):
        extract_project_documents(project, [pdf])
