import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "index.html"
APP = ROOT / "web" / "app.js"
SOURCE_STATE = ROOT / "web" / "source-state.js"
STYLES = ROOT / "web" / "styles.css"


def test_source_aware_recommendations_markup_and_scripts_are_wired() -> None:
    index = INDEX.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert 'data-page="compare" type="button">Recommendations<' in index
    for element_id in (
        "recommendationSource",
        "recommendationDataset",
        "recommendationGroundtruth",
        "recommendationOfficialSet",
        "recommendationProject",
        "recommendationRun",
        "recommendationContext",
        "recommendationStrip",
        "recommendationTable",
        "pricingLedger",
        "tradeoffDetails",
        "projectEvidenceDialog",
    ):
        assert f'id="{element_id}"' in index

    assert "Dataset" in index
    assert "Ground truth" in index
    assert "Official benchmark ground truth" in index
    assert "No completed run for this dataset and ground truth" in app

    assert "Official reranked matrix — 180" in index
    assert "No-reranker baseline — measured results" in index
    assert '<dialog id="projectEvidenceDialog" class="evidence-dialog" aria-labelledby="projectEvidenceTitle" aria-describedby="projectEvidenceStatus">' in index
    assert "Average retrieval plus reranking latency per query" in index
    assert 'href="/styles.css?v=20260716-meeting-hardening"' in index
    assert 'src="/recommendations.js?v=20260716-extraction-override"' in index
    assert index.index('/recommendations.js?v=20260716-extraction-override') < index.index(
        '/app.js?v=20260716-meeting-hardening'
    )

    for endpoint in (
        "/api/result-sources",
        "/api/project-runs?",
        "/api/project-run-results?",
        "/api/project-run-evidence?",
    ):
        assert endpoint in app
    assert "AbortController" in app
    assert "configured combinations" in app


def test_recommendations_styles_are_responsive_and_use_existing_tokens() -> None:
    styles = STYLES.read_text(encoding="utf-8")

    for selector in (
        ".result-source-bar",
        ".recommendation-strip",
        ".recommendation-item",
        ".pricing-ledger",
        ".tradeoff-details",
        ".evidence-dialog",
    ):
        assert selector in styles
    assert "@media(max-width:760px)" in styles.replace(" ", "")
    assert "var(--mint)" in styles
    assert "var(--accent)" in styles
    assert "var(--amber)" in styles
    assert ".result-source-bar label[hidden]" in styles


def test_late_project_run_response_cannot_overwrite_newer_selection() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const pending = new Map();
const elements = {{}};
function element(id) {{
  return elements[id] ||= {{
    id, value: '', textContent: '', innerHTML: '', hidden: false,
    options: [], selectedOptions: [],
    classList: {{toggle(){{}}, add(){{}}, remove(){{}}}},
    addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}},
    showModal(){{}}, close(){{}},
  }};
}}
const context = {{
  console,
  AbortController,
  URLSearchParams,
  PipelineRecommendations: {{
    recommendationsForSource(source) {{
      return {{mode: source.scoring_mode === 'retrieval_labels' ? 'labelled' : 'evidence_only', roles: {{}}, completed: source.rows || [], failed: [], rows: source.rows || []}};
    }},
    pricingLedger() {{return []; }},
    metricLabels() {{return {{}};}},
    rowBadges() {{return []; }},
    pricingForRow() {{return {{state:'no_api_fee'}};}},
  }},
  document: {{
    getElementById: element,
    querySelectorAll() {{return []; }},
    addEventListener() {{}},
    body: {{insertAdjacentHTML() {{}}}},
  }},
  window: {{}},
  location: {{hash:'#compare'}},
  history: {{replaceState() {{}}}},
  setTimeout,
  clearTimeout,
  fetch(url) {{
    return new Promise(resolve => pending.set(url, payload => resolve({{ok:true, json:async()=>payload, text:async()=>JSON.stringify(payload)}})));
  }},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source + `
  globalThis.__loadProjectRun = loadProjectRun;
  globalThis.__recommendationState = recommendationState;
`, context);
(async () => {{
  const first = context.__loadProjectRun('project_a', 'run_a');
  const second = context.__loadProjectRun('project_b', 'run_b');
  pending.get('/api/project-run-results?project_id=project_b&run_id=run_b')({{
    source_type:'uploaded_project', project_id:'project_b', project_label:'Beta', run_id:'run_b',
    scoring_mode:'evidence_only', metric_k:10, rows:[{{combo_id:'beta',status:'completed'}}],
  }});
  await second;
  pending.get('/api/project-run-results?project_id=project_a&run_id=run_a')({{
    source_type:'uploaded_project', project_id:'project_a', project_label:'Alpha', run_id:'run_a',
    scoring_mode:'evidence_only', metric_k:10, rows:[{{combo_id:'alpha',status:'completed'}}],
  }});
  await first;
  const active = context.__recommendationState.active;
  if (!active || active.project_id !== 'project_b' || active.run_id !== 'run_b') {{
    console.error(JSON.stringify(active));
    process.exit(1);
  }}
  if (!elements.recommendationContext.textContent.includes('Beta')) {{
    console.error(elements.recommendationContext.textContent);
    process.exit(1);
  }}
}})().catch(error => {{console.error(error); process.exit(1);}});
"""
    proc = subprocess.run(
        ["node", "-e", script],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_misbound_project_run_response_is_rejected() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const pending = new Map();
const elements = {{}};
function element(id) {{
  return elements[id] ||= {{
    id, value: '', textContent: '', innerHTML: '', hidden: false,
    options: [], selectedOptions: [],
    classList: {{toggle(){{}}, add(){{}}, remove(){{}}}},
    addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}},
    showModal(){{}}, close(){{}},
  }};
}}
const context = {{
  console, AbortController, URLSearchParams,
  PipelineRecommendations: {{
    recommendationsForSource(source) {{return {{mode:'evidence_only',roles:{{}},completed:source.rows||[],failed:[],rows:source.rows||[]}};}},
    pricingLedger() {{return []; }}, metricLabels() {{return {{}};}}, rowBadges() {{return []; }},
    pricingForRow() {{return {{state:'no_api_fee'}};}},
  }},
  document: {{getElementById:element,querySelectorAll(){{return[];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},
  window: {{}}, location: {{hash:'#compare'}}, history: {{replaceState(){{}}}}, setTimeout, clearTimeout,
  fetch(url) {{return new Promise(resolve => pending.set(url, payload => resolve({{ok:true,json:async()=>payload,text:async()=>JSON.stringify(payload)}})));}},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source + `
  globalThis.__loadProjectRun = loadProjectRun;
  globalThis.__recommendationState = recommendationState;
`, context);
(async () => {{
  const request = context.__loadProjectRun('project_a', 'run_a');
  pending.get('/api/project-run-results?project_id=project_a&run_id=run_a')({{
    source_type:'uploaded_project', project_id:'project_b', project_label:'Wrong project', run_id:'run_b',
    scoring_mode:'evidence_only', metric_k:10, rows:[{{combo_id:'wrong',status:'completed'}}],
  }});
  await request;
  if (context.__recommendationState.active !== null) {{
    console.error(JSON.stringify(context.__recommendationState.active));
    process.exit(1);
  }}
  if (!elements.recommendationContext.textContent.includes('unavailable')) {{
    console.error(elements.recommendationContext.textContent);
    process.exit(1);
  }}
}})().catch(error => {{console.error(error);process.exit(1);}});
"""
    proc = subprocess.run(
        ["node", "-e", script], text=True, capture_output=True, timeout=10
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stale_source_catalog_cannot_overwrite_official_selection() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const pending = new Map();
const elements = {{}};
function element(id) {{
  return elements[id] ||= {{
    id, value: '', textContent: '', innerHTML: '', hidden: false, open: false,
    selectedOptions: [], classList: {{toggle(){{}}}}, addEventListener(){{}},
    querySelectorAll(){{return [];}}, setAttribute(){{}}, showModal(){{}}, close(){{}},
  }};
}}
const context = {{
  console, AbortController, URLSearchParams,
  PipelineRecommendations: {{
    recommendationsForSource(source) {{return {{mode:'labelled',roles:{{}},completed:source.rows||[],failed:[],rows:source.rows||[]}};}},
    pricingLedger() {{return [];}}, metricLabels() {{return {{}};}}, rowBadges() {{return [];}},
    pricingForRow() {{return {{state:'no_api_fee'}};}},
  }},
  document: {{getElementById:element,querySelectorAll(){{return[];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},
  window: {{}}, location: {{hash:'#compare'}}, history: {{replaceState(){{}}}}, setTimeout, clearTimeout,
  fetch(url) {{return new Promise(resolve => pending.set(url, payload => resolve({{ok:true,json:async()=>payload,text:async()=>JSON.stringify(payload)}})));}},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source + `
  globalThis.__selectRecommendationSource = selectRecommendationSource;
  globalThis.__recommendationState = recommendationState;
`, context);
(async () => {{
  const uploaded = context.__selectRecommendationSource('uploaded_project');
  await Promise.resolve();
  await context.__selectRecommendationSource('official');
  pending.get('/api/result-sources')({{official:{{source_type:'official',configured:180,evaluated:2}},projects:[]}});
  await uploaded;
  if (context.__recommendationState.sourceType !== 'official') process.exit(1);
  const text = elements.recommendationContext.textContent;
  if (!text.includes('180 configured combinations') || text.includes('No uploaded projects')) {{
    console.error(text);
    process.exit(1);
  }}
}})().catch(error => {{console.error(error);process.exit(1);}});
"""
    proc = subprocess.run(
        ["node", "-e", script], text=True, capture_output=True, timeout=10
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_shared_embedding_cost_is_not_rendered_as_zero_per_combination() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const R = require({str(ROOT / 'web' / 'recommendations.js')!r});
function element() {{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const context = {{
  console,
  PipelineRecommendations:R,
  AbortController,
  URLSearchParams,
  document:{{getElementById(){{return element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},
  window:{{}}, location:{{hash:'#overview'}}, history:{{replaceState(){{}}}}, setTimeout(){{}},
  fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source + `
  const row={{
    combo_id:'shared', commercial_model_ids:['openai_text-embedding-3-large'],
    measured_usage:{{embedding_input_tokens:1000,embedding_usage_scope:'shared_embedding',embedding_usage_key:'chunk|openai'}},
  }};
  globalThis.__pricingText=recommendationPricingText(PipelineRecommendations.pricingForRow(row));
`, context);
if (context.__pricingText !== 'Shared embedding cost · see pricing ledger') {{
  console.error(context.__pricingText);
  process.exit(1);
}}
"""
    proc = subprocess.run(
        ["node", "-e", script],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_evidence_only_aggregate_roles_have_decision_useful_titles() -> None:
    script = f"""
const fs=require('fs'), vm=require('vm');
function element() {{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(){{return element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
  globalThis.__titles=[
    recommendationRoleTitle('fastest',{{state:'latency_not_recorded'}}),
    recommendationRoleTitle('run_health',{{state:'run_health',succeeded:1,failed:1,total:2}}),
    recommendationRoleTitle('evidence_coverage',{{state:'evidence_coverage',evidence_count:4,query_count:3}}),
    recommendationRoleTitle('evidence_coverage',{{state:'evidence_coverage',evidence_count:null,query_count:3}}),
  ];
`,context);
const expected=['Latency not recorded','1/2 succeeded','4 evidence rows','Evidence not recorded'];
if (JSON.stringify(context.__titles)!==JSON.stringify(expected)) {{console.error(JSON.stringify(context.__titles));process.exit(1);}}
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_recommendation_rendering_escapes_remote_labels_and_keeps_metric_fallbacks() -> None:
    script = f"""
const fs=require('fs'), vm=require('vm');
function element() {{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const ids=['recommendationStatus','recommendationContext','recommendationStrip','recommendationTable','pricingLedger','recommendationModeNote'];
const elements=Object.fromEntries(ids.map(id=>[id,element()]));
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(id){{return elements[id]||element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
  renderRecommendationSource({{
    source_type:'uploaded_project',project_id:'p',project_label:'<img src=x onerror=1>',run_id:'run_x',run_state:'completed',
    scoring_mode:'retrieval_labels',metric_k:10,rows:[
      {{combo_id:'c',status:'completed',chunker_id:'<img src=x onerror=1>',embedding_id:'e',vector_store_id:'s',reranker_id:'r',labelled_queries:1,recall_at_k:.5,mrr_at_k:null,ndcg_at_k:null,avg_query_latency_s:null,commercial_model_ids:[],measured_usage:{{}}}},
      {{combo_id:'no-labels',status:'completed',chunker_id:'safe',embedding_id:'e',vector_store_id:'s',reranker_id:'r',labelled_queries:0,recall_at_k:null,mrr_at_k:null,ndcg_at_k:null,avg_query_latency_s:.2,commercial_model_ids:[],measured_usage:{{}}}},
    ],
  }});
  globalThis.__labelled=document.getElementById('recommendationTable').innerHTML;
  renderRecommendationSource({{
    source_type:'uploaded_project',project_id:'p',project_label:'safe',run_id:'run_y',run_state:'completed',
    scoring_mode:'evidence_only',metric_k:10,evidence_counts_by_combo:{{d:null}},rows:[{{combo_id:'d',status:'completed',chunker_id:'safe',embedding_id:'e',vector_store_id:'s',reranker_id:'r',query_count:3,labelled_queries:0,recall_at_k:null,mrr_at_k:null,ndcg_at_k:null,avg_query_latency_s:.1,evidence_count:null,commercial_model_ids:[],measured_usage:{{}}}}],
  }});
  globalThis.__evidence=document.getElementById('recommendationTable').innerHTML;
`,context);
if (context.__labelled.includes('<img') || !context.__labelled.includes('&lt;img') || !context.__labelled.includes('Not recorded') || !context.__labelled.includes('Not applicable')) {{console.error(context.__labelled);process.exit(1);}}
if (!context.__evidence.includes('<th>Queries</th>') || context.__evidence.includes('Recall@') || context.__evidence.includes('<th>MRR</th>') || !context.__evidence.includes('Not recorded')) {{console.error(context.__evidence);process.exit(1);}}
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_mixed_commercial_cost_text_and_ledger_distinguish_shared_and_combination_scope() -> None:
    script = f"""
const fs=require('fs'),vm=require('vm');
const elements={{pricingLedger:{{innerHTML:'',textContent:'',classList:{{toggle(){{}}}},addEventListener(){{}}}}}};
function element(){{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(id){{return elements[id]||(elements[id]=element());}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
  const row={{combo_id:'mixed',chunker_id:'chunk',embedding_id:'openai_text-embedding-3-large',vector_store_id:'FAISS',reranker_id:'Amazon Rerank v1',commercial_model_ids:['openai_text-embedding-3-large','Amazon Rerank v1'],measured_usage:{{embedding_input_tokens:1000000,embedding_usage_scope:'shared_embedding',embedding_usage_key:'chunk|openai',rerank_search_units:7,rerank_usage_scope:'combination'}}}};
  globalThis.__text=recommendationPricingText(PipelineRecommendations.pricingForRow(row));
  renderPricingLedger({{source_type:'uploaded_project',rows:[row]}});
  globalThis.__ledger=document.getElementById('pricingLedger').innerHTML;
`,context);
if (context.__text !== 'Shared embedding cost · see pricing ledger') {{console.error(context.__text);process.exit(1);}}
if (!context.__ledger.includes('$0.130000 measured run cost') || !context.__ledger.includes('$0.007000 measured combination cost')) {{console.error(context.__ledger);process.exit(1);}}
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_recommendation_table_is_paginated_and_explains_winner_badges() -> None:
    script = f"""
const fs=require('fs'), vm=require('vm');
function element() {{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const ids=['recommendationStatus','recommendationContext','recommendationStrip','recommendationTable','recommendationSort','recommendationFilters','pricingLedger','recommendationModeNote'];
const elements=Object.fromEntries(ids.map(id=>[id,element()]));
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(id){{return elements[id]||element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const rows=Array.from({{length:12}},(_,i)=>({{
  combo_id:`combo-${{i}}`,status:'completed',chunker_id:`chunk-${{i}}`,embedding_id:'e',vector_store_id:'s',reranker_id:'r',
  labelled_queries:1,recall_at_k:1-i/20,mrr_at_k:1-i/20,ndcg_at_k:1-i/20,avg_query_latency_s:.1+i,
  commercial_model_ids:[],measured_usage:{{}},evidence_count:1,
}}));
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
  renderRecommendationSource({{source_type:'uploaded_project',project_id:'p',project_label:'safe',run_id:'run_x',run_state:'completed',scoring_mode:'retrieval_labels',metric_k:10,rows:${{JSON.stringify(rows)}}}});
  globalThis.__table=document.getElementById('recommendationTable').innerHTML;
  globalThis.__pager=document.getElementById('recommendationSort').innerHTML;
`,context);
const bodyRows=(context.__table.match(/<tr/g)||[]).length-1;
if (bodyRows!==10 || !context.__pager.includes('Next 10') || !context.__pager.includes('1–10 of 12')) {{console.error(context.__table,context.__pager);process.exit(1);}}
if (!context.__table.includes('#1') || !context.__table.includes('Quality') || !context.__table.includes('Speed') || !context.__table.includes('No API fee') || !context.__table.includes('Why it stands out')) {{console.error(context.__table);process.exit(1);}}
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    app = APP.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")
    assert 'data-label="Pipeline"' in app
    assert "Recommendation mobile cards" in styles
    assert "#recommendationTable td::before" in styles


def test_evidence_paging_reports_cumulative_loaded_rows() -> None:
    script = f"""
const fs=require('fs'),vm=require('vm');
function element(){{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const elements={{projectEvidenceStatus:element(),projectEvidenceRows:element(),projectEvidenceMore:element()}};
const first=Array.from({{length:25}},(_,i)=>({{combo_id:'combo',query_id:`q${{i}}`,query:'Q',source_name:'safe.txt',excerpt:'E',rank:i+1}}));
const second=Array.from({{length:5}},(_,i)=>({{combo_id:'combo',query_id:`q${{i+25}}`,query:'Q',source_name:'safe.txt',excerpt:'E',rank:i+26}}));
const pages=[{{rows:first,next_offset:25}},{{rows:second,next_offset:null}}];
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(id){{return elements[id]||element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>pages.shift(),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
(async()=>{{
  recommendationState.active={{source_type:'uploaded_project',project_id:'p',run_id:'run_x'}};
  recommendationState.evidenceComboId='combo';
  await loadProjectEvidencePage();
  await loadProjectEvidencePage({{append:true}});
  globalThis.__status=document.getElementById('projectEvidenceStatus').textContent;
  globalThis.__offset=recommendationState.evidenceOffset;
}})().catch(error=>{{console.error(error);process.exitCode=1;}});
`,context);
setTimeout(()=>{{
  if(context.__status!=='30 evidence rows loaded'||context.__offset!==0){{console.error(context.__status,context.__offset);process.exit(1);}}
}},20);
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_evidence_paging_rejects_non_advancing_cursor_without_duplicate_rows() -> None:
    script = f"""
const fs=require('fs'),vm=require('vm');
function element(){{return {{value:'',textContent:'',innerHTML:'',hidden:false,open:false,selectedOptions:[],classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const elements={{projectEvidenceStatus:element(),projectEvidenceRows:element(),projectEvidenceMore:element()}};
const row={{combo_id:'combo',query_id:'q1',query:'Q',source_name:'safe.txt',excerpt:'E',rank:1}};
const pages=[{{rows:[row],next_offset:25}},{{rows:[row],next_offset:25}}];
const context={{console,AbortController,URLSearchParams,PipelineRecommendations:require({str(ROOT / 'web' / 'recommendations.js')!r}),document:{{getElementById(id){{return elements[id]||element();}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>pages.shift(),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const source=fs.readFileSync({str(APP)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source+`
(async()=>{{
  recommendationState.active={{source_type:'uploaded_project',project_id:'p',run_id:'run_x'}};
  recommendationState.evidenceComboId='combo';
  await loadProjectEvidencePage();
  await loadProjectEvidencePage({{append:true}});
  globalThis.__status=document.getElementById('projectEvidenceStatus').textContent;
  globalThis.__count=recommendationState.evidenceLoadedCount;
  globalThis.__html=document.getElementById('projectEvidenceRows').innerHTML;
  globalThis.__more=document.getElementById('projectEvidenceMore').hidden;
}})().catch(error=>{{console.error(error);process.exitCode=1;}});
`,context);
setTimeout(()=>{{
  const rendered=(context.__html.match(/evidence-dialog-row/g)||[]).length;
  if(context.__status!=='Evidence unavailable for this combination'||context.__count!==1||rendered!==1||context.__more!==true){{console.error(context.__status,context.__count,rendered,context.__more);process.exit(1);}}
}},20);
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_official_recommendations_use_only_official_rows_and_recorded_evidence() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function element(id) {{
  return elements[id] ||= {{
    id, value: '', textContent: '', innerHTML: '', hidden: false,
    options: [], selectedOptions: [],
    classList: {{toggle(){{}}, add(){{}}, remove(){{}}}},
    addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}},
    showModal(){{}}, close(){{}},
  }};
}}
const context = {{
  console, AbortController, URLSearchParams,
  PipelineRecommendations: {{}},
  document: {{getElementById:element,querySelectorAll(){{return[];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},
  window: {{}}, location: {{hash:'#compare'}}, history: {{replaceState(){{}}}}, setTimeout, clearTimeout,
  fetch() {{throw new Error('unexpected fetch');}},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const source = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(source + `
  const faiss = {{sheet:'Heading_sections_l2',embedding:'gte_multilingual_base',store:'FAISS',reranker:'bge-reranker-base',recall_at_5:0.964,mrr:0.888433,ndcg_at_5:0.902323,avg_latency_seconds:0.024932915}};
  const qdrant = {{sheet:'Heading_sections_l2',embedding:'gte_multilingual_base',store:'Qdrant',reranker:'bge-reranker-base',recall_at_5:0.90,mrr:0.80,ndcg_at_5:0.81,avg_latency_seconds:0.01}};
  state = {{operational:{{
    evaluation:{{
      summary:[{{sheet:'legacy',embedding:'gte_multilingual_base',store:'Qdrant',reranker:'none',winner_score:1,recall_at_5:1}}],
      reranked:{{summary:[]}},
      benchmark_reference:{{summary:[faiss,qdrant]}},
    }},
    benchmark_detail_evidence:[faiss,faiss],
  }}}};
  recommendationState.official = {{configured:180,evaluated:2}};
  globalThis.__payload = {{
    official: officialRecommendationPayload('official'),
    baseline: officialRecommendationPayload('baseline'),
  }};
`, context);
console.log(JSON.stringify(context.__payload));
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payloads = json.loads(proc.stdout)
    payload = payloads["official"]
    assert len(payload["rows"]) == 2
    assert payload["result_set"] == "official"
    assert all(row["reranker"] == "bge-reranker-base" for row in payload["rows"])
    faiss = next(row for row in payload["rows"] if row["store"] == "FAISS")
    qdrant = next(row for row in payload["rows"] if row["store"] == "Qdrant")
    assert faiss["winner_score"] > qdrant["winner_score"]
    assert faiss["evidence_count"] == 2
    baseline = payloads["baseline"]
    assert baseline["result_set"] == "baseline"
    assert len(baseline["rows"]) == 1
    assert baseline["rows"][0]["reranker"] == "none"
    assert baseline["rows"][0]["sheet"] == "legacy"
