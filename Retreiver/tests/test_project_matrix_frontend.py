from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


def test_uploaded_project_pipeline_uses_isolated_json_matrix_routes() -> None:
    assert "function projectMatrixPayload" in APP
    assert "'/api/run/preflight-project-matrix'" in APP
    assert "'/api/run/project-matrix'" in APP
    assert "headers: {'Content-Type': 'application/json'}" in APP
    assert "JSON.stringify(payload)" in APP
    assert "datasetId.startsWith('project:')" in APP
    assert "Project evidence-only mode" in APP
    assert "Optional reranking does not create quality metrics" in APP


def test_project_matrix_payload_uses_ids_queries_and_component_selections_only() -> None:
    required_fragments = (
        "dataset_id: datasetId",
        "groundtruth_id: groundtruthId",
        "typed_queries: evidenceOnly ? typedRunQueries() : []",
        "chunkers: projectSelectedValues('runSheet'",
        "embeddings: projectSelectedValues('runEmbedding'",
        "vector_stores: projectSelectedValues('runStore'",
        "rerankers: projectSelectedRerankers()",
        "top_k: Number($('runRetrievalTopK')?.value || 10)",
    )
    for fragment in required_fragments:
        assert fragment in APP
    assert "workbook" not in APP[APP.index("function projectMatrixPayload"):APP.index("function projectMatrixPayload") + 1800]
    assert "groundtruth_path" not in APP


def test_default_dataset_keeps_legacy_complete_pipeline_route() -> None:
    assert "/api/run/preflight-complete-pipeline" in APP
    assert "/api/run/complete-pipeline" in APP
    assert "if (isProjectDatasetSelected())" in APP


def test_project_matrix_payload_executes_with_canonical_ids_and_no_reranker() -> None:
    app_path = ROOT / "web" / "app.js"
    script = f"""
const fs = require('fs'), vm = require('vm');
const generic = () => ({{value:'', selectedOptions:[], options:[], textContent:'', innerHTML:'',
  classList:{{toggle(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}});
const field = (value='', selectedOptions=[]) => Object.assign(generic(), {{value, selectedOptions}});
const elements = {{
  runDataset: field('project:demo'),
  runGroundtruth: field('groundtruth:none'),
  runQueries: field(' First question? \\n\\nSecond question? '),
  runRetrievalTopK: field('7'),
  runSheet: field('', [{{value: 'all'}}]),
  runEmbedding: field('', [{{value: 'jina_v3'}}]),
  runStore: field('', [{{value: 'FAISS'}}]),
  runRerankerMain: field('', [{{value: 'none'}}]),
}};
const context = {{console, document:{{getElementById(id){{return elements[id] || (elements[id]=generic());}},
  querySelectorAll(){{return [];}}, addEventListener(){{}}, body:{{insertAdjacentHTML(){{}}}}}},
  window:{{}}, location:{{hash:'#overview'}}, history:{{replaceState(){{}}}},
  fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}), setTimeout(){{}}}};
vm.createContext(context);
const code = fs.readFileSync({str(app_path)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
benchmarkOptions={{chunkers:['fixed_tok1200_ov150','entity_heuristic_w6'],embeddings:['jina_v3'],vector_stores:['FAISS'],rerankers:['Qwen3:4B Rerank']}};
globalThis.__payload=projectMatrixPayload();
`, context);
process.stdout.write(JSON.stringify(context.__payload));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "dataset_id": "project:demo",
        "groundtruth_id": "groundtruth:none",
        "typed_queries": ["First question?", "Second question?"],
        "top_k": 7,
        "selections": {
            "chunkers": ["fixed_tok1200_ov150", "entity_heuristic_w6"],
            "embeddings": ["jina_v3"],
            "vector_stores": ["FAISS"],
            "rerankers": [],
        },
        "large_matrix_confirmation": None,
    }
