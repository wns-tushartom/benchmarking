import json
import subprocess
from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "web" / "app.js"


def test_successful_upload_selects_the_new_project_and_exits_evidence_only_mode():
    project = {
        "project_id": "public-five-paper_123",
        "label": "Public five-paper",
        "chunk_count": 1573,
        "groundtruth": {"valid": True, "reason": ""},
        "setup": {"state": "not_started", "stages": []},
    }
    js = f"""
const fs = require('fs'), vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  return {{
    id, value: '', textContent: '', innerHTML: '', files: [], selectedOptions: [], listeners: {{}},
    classList: {{ toggle() {{}}, contains() {{ return false; }} }},
    addEventListener(name, handler) {{ this.listeners[name] = handler; }},
    setAttribute() {{}}, closest() {{ return null; }},
  }};
}}
function el(id) {{ return elements[id] ||= makeElement(id); }}
class FakeFormData {{ append() {{}} }}
const context = {{
  console, FormData: FakeFormData,
  document: {{ getElementById: el, querySelectorAll() {{ return []; }}, addEventListener() {{}}, body: {{ insertAdjacentHTML() {{}} }} }},
  window: {{}}, location: {{hash:'#upload'}}, history: {{replaceState() {{}}}}, setTimeout() {{}},
  fetch: async (url) => url === '/api/upload-dataset'
    ? ({{ok:true, json:async()=>({json.dumps(project)})}})
    : ({{ok:true, json:async()=>({{}}), text:async()=>''}}),
}};
vm.createContext(context);
const code = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code, context);
vm.runInContext('state = ' + JSON.stringify({{operational: {{user_projects: [{json.dumps(project)}]}}}}) + ';', context);
context.refresh = async () => context.renderUserProjects([{json.dumps(project)}]);
el('datasetFile').files = [{{name: 'corpus.zip'}}];
el('datasetLabel').value = 'Public five-paper';
el('projectQueryStatus').textContent = 'Evidence-only mode';
(async () => {{
  await context.uploadDataset({{preventDefault() {{}}}});
  if (el('projectSelect').value !== 'public-five-paper_123') throw new Error('new project was not selected');
  if (el('globalDataset').value !== 'public-five-paper_123') throw new Error('global dataset control did not mirror the selected project');
  if (!el('globalGroundtruth').innerHTML.includes('Linked ground truth')) throw new Error('global linked-ground-truth control was not refreshed');
  if (!el('globalGroundtruth').disabled) throw new Error('project ground truth must be derived and locked');
  if (!el('runGroundtruthStatus').textContent.includes('Ground truth linked')) throw new Error('ground-truth state was not refreshed');
  if (el('runGroundtruthStatus').textContent.includes('Evidence-only')) throw new Error('stale evidence-only state remained visible');
  if (!el('projectQueryStatus').textContent.includes('Ground truth linked') || el('projectQueryStatus').textContent.includes('Evidence-only')) throw new Error('project query badge was not updated');
}})().catch(error => {{ console.error(error.stack); process.exit(1); }});
"""
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_global_dataset_and_linked_groundtruth_controls_exist():
    index = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="globalDataset"' in index
    assert 'id="globalGroundtruth"' in index
    assert 'id="globalSourceContext"' in index
