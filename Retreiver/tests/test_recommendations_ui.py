import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "index.html"
APP = ROOT / "web" / "app.js"


def test_recommendations_tab_and_assets_are_present() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert 'data-page="compare" type="button">Recommendations</button>' in html
    assert 'id="recommendationStrip"' in html
    assert 'id="recommendationTable"' in html
    assert 'id="pricingLedger"' in html
    assert '/recommendations.js?v=20260715-recommendations-restored' in html
    assert '/app.js?v=20260715-recommendations-restored' in html
    assert html.index('/recommendations.js?') < html.index('/app.js?')
    assert "Accuracy, cost, latency</button>" not in html


def test_official_recommendations_render_quality_speed_and_rows() -> None:
    script = f"""
const fs = require('fs'), vm = require('vm');
function element() {{
  return {{
    value:'official', textContent:'', innerHTML:'', hidden:false, open:false,
    classList:{{toggle(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}
  }};
}}
const ids = ['recommendationStatus','recommendationContext','recommendationStrip',
  'recommendationTable','pricingLedger','recommendationModeNote','recommendationSort',
  'top10ScoreChart','qualityLatencyChart','comparisonGrid','rerankerLiftChart','stageComparisonChart'];
const elements = Object.fromEntries(ids.map(id => [id, element()]));
const context = {{
  console,
  elements,
  PipelineRecommendations: require({str(ROOT / 'web' / 'recommendations.js')!r}),
  document: {{
    getElementById(id) {{ return elements[id] || element(); }},
    querySelectorAll() {{ return []; }},
    addEventListener() {{}},
    body: {{insertAdjacentHTML() {{}}}},
  }},
  globalThis: null,
  location: {{hash:'#compare'}},
  history: {{replaceState() {{}}}},
  fetch: async () => ({{ok:true,json:async()=>({{}}),text:async()=>''}}),
  setTimeout() {{}},
}};
context.globalThis = context;
vm.createContext(context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split("$('refreshBtn')?.addEventListener")[0];
vm.runInContext(source + `
renderRecommendationSource({{
  source_type:'official', scoring_mode:'retrieval_labels', result_set:'official',
  configured:2, evaluated:2, rows:[
    {{combo_id:'quality',sheet:'semantic',embedding:'jina',store:'qdrant',reranker:'qwen',winner_score:.91,recall_at_5:.9,mrr:.8,ndcg_at_5:.85,avg_latency_seconds:.4}},
    {{combo_id:'speed',sheet:'fixed',embedding:'gte',store:'faiss',reranker:'none',winner_score:.7,recall_at_5:.75,mrr:.65,ndcg_at_5:.68,avg_latency_seconds:.1}},
  ]
}});
globalThis.__out = {{
  strip: elements.recommendationStrip.innerHTML,
  table: elements.recommendationTable.innerHTML,
  status: elements.recommendationStatus.textContent,
}};
`, context);
const out = context.__out;
if (!out.strip.includes('Quality') || !out.strip.includes('Speed') || !out.strip.includes('Value')) process.exit(1);
if (!out.table.includes('semantic') || !out.table.includes('fixed')) process.exit(2);
if (out.status !== 'Official benchmark') process.exit(3);
"""
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
