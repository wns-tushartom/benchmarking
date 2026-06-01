# WNS VM artifact sync and dashboard wiring

Current VM prompt shown by user:

```text
U481019@WNSAWSAIRESE-06:~$ ls
benchmarking
```

Assumption: the repository root is one of:

```text
~/benchmarking/Retreiver
~/benchmarking
```

## 1. Copy updated files to the VM

From the updated ZIP, replace these files in the VM repo:

```text
Retreiver/scripts/collect_vm_dashboard_artifacts.py
Retreiver/scripts/run_retrieval_smoke_from_vm_dbs.py
Retreiver/scripts/serve_benchmark_dashboard.py
Retreiver/web/index.html
Retreiver/web/app.js
Retreiver/web/styles.css
```

## 2. On the VM, go to the Retreiver directory

Run:

```bash
cd ~/benchmarking/Retreiver || cd ~/benchmarking
pwd
```

If `scripts/run_long_db_ingestion.py` exists in the current directory, you are in the right place.

Check:

```bash
ls scripts/run_long_db_ingestion.py data
```

## 3. Confirm artifacts currently exist

Run:

```bash
find data/db_ingestion_runs -maxdepth 2 -type f -name 'summary.csv' -print 2>/dev/null
find data/embedding_cache -maxdepth 1 -type f -name '*_meta.json' -print 2>/dev/null | head
find data/reranker_smoke -maxdepth 1 -type f -name '*.json' -print 2>/dev/null
```

Expected: at least one `summary.csv` under `data/db_ingestion_runs`.

## 4. Run live retrieval smoke tests from the existing DB collections

This tests actual persisted vector DB collections/tables/classes discovered from the ingestion `summary.csv` files.

Recommended first run:

```bash
python3 scripts/run_retrieval_smoke_from_vm_dbs.py \
  --queries "refund old ticket and issue new ticket" "schedule change alternate option" "NACO refund process" \
  --sheets fixed_tok1200_ov150 Heading_sections_l2 semantic_split \
  --embeddings gte_multilingual_base jina_v3 \
  --stores Qdrant PGVector Weaviate \
  --top-k 5 \
  --max-combos 18
```

This writes:

```text
data/retrieval_smoke/*.json
data/retrieval_smoke/summary.json
```

If it fails because a DB/env variable is missing, check `.env` in the repo root. It should include roughly:

```bash
GTE_EMBEDDING_URL=http://127.0.0.1:5000/embed/gte
JINA_EMBEDDING_URL=http://127.0.0.1:5000/embed/jina
QDRANT_URL=http://127.0.0.1:5001
PGVECTOR_DSN=postgresql://wns:wns_password@127.0.0.1:5003/wns_benchmark
WEAVIATE_URL=http://127.0.0.1:5004
```

## 5. Create dashboard artifact bundle on VM

Run:

```bash
python3 scripts/collect_vm_dashboard_artifacts.py
```

It prints a path like:

```text
zip=/home/U481019/wns_dashboard_artifacts_YYYYMMDD_HHMMSS.zip
```

This ZIP contains the operational artifacts the dashboard needs:

```text
data/db_ingestion_runs/*/summary.csv
data/retrieval_smoke/*.json
data/reranker_smoke/*.json
data/embedding_cache/*_meta.json
data/vm_dashboard_snapshot.json
WNS_VM_PROGRESS_*.md
JINA_TIMINGS_*.md
RERANKER_SMOKE_*.md
```

## 6. Move bundle from VM to dashboard machine

Option A, Jupyter:

1. Start/open the same Jupyter bridge used earlier.
2. Download `~/wns_dashboard_artifacts_*.zip` from the VM home directory.
3. Upload it here or to the dashboard machine.

Option B, terminal copy-paste if Jupyter is not convenient:

```bash
base64 -w 0 ~/wns_dashboard_artifacts_YYYYMMDD_HHMMSS.zip > ~/wns_dashboard_artifacts.b64
```

Then copy the base64 text out. On the receiving machine:

```bash
base64 -d wns_dashboard_artifacts.b64 > wns_dashboard_artifacts.zip
```

## 7. Apply artifact bundle on dashboard machine

From the dashboard repo root:

```bash
cd Retreiver
unzip -o /path/to/wns_dashboard_artifacts_YYYYMMDD_HHMMSS.zip
python3 scripts/serve_benchmark_dashboard.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

Verify API:

```bash
curl -s http://127.0.0.1:8765/api/results | python3 -m json.tool | head -80
```

Expected JSON should contain:

```text
operational.ingestion.rows
operational.retrieval_smokes
operational.reranker_smokes
operational.service_health
```

## 8. What the updated frontend now shows

The dashboard now reads and displays:

```text
DB ingestion combos
latest ingestion rows
actual collection/table/class names
VM service health snapshot
live retrieval smoke tests from Qdrant/PGVector/Weaviate
reranker smoke tests
artifact list
```

Recall/MRR/nDCG remains intentionally separate until the ground-truth file is available.
