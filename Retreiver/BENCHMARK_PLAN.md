# WNS-AIR RAG Retrieval Benchmark Plan

**Document Created:** 2026-05-13  
**Repository Path:** `General_Components/QA_Text/Retreiver`  
**Purpose:** Systematic benchmarking of RAG retrieval pipeline configurations

⚠️ **IMPORTANT:** This is a planning document. All code examples, scripts, classes, and sample outputs are **PROPOSED** and not yet implemented unless explicitly marked as existing.

---

## Executive Summary

This plan outlines the implementation of a comprehensive benchmarking framework to evaluate combinations of:
- **Chunking Methods** (4 options - to be implemented)
- **Embedding Models** (3 options - to be implemented)
- **Vector Databases** (4 options - 1 exists, 3 to be implemented)
- **Index Types** (1 option: HNSW - to be configured)
- **Retrieval Methods** (1 option: Cosine Similarity - exists)
- **Reranking Models** (3 options - to be implemented)

**Official Benchmark Matrix:** 4 × 3 × 4 × 1 × 1 × 3 = **144 configurations**

**Engineering Baseline:** Add "no reranking" option for comparison:
- 4 × 3 × 4 × 1 × 1 × 4 = **192 configurations** (includes no_reranker baseline)
- Minimum: Run current baseline + no_reranker separately

**Proposed Metrics** (to be confirmed):
- **Retrieval Quality:** Precision, Recall, MRR, NDCG (to be implemented)
- **Answer Quality:** ROUGE, BERTScore, BLEURT, Cosine Similarity (partially exists)
- **Performance:** Latency (indexing, retrieval, end-to-end) (to be implemented)
- **Cost:** API calls, compute time (to be tracked)

---

## Benchmark Matrix

### Dimension 1: Chunking Methods (4 options)

| Method | Description | Implementation Status | Complexity |
|--------|-------------|----------------------|------------|
| `entity_heuristic_w4` | Entity-based chunking, window size 4 | ❌ Not implemented | Medium |
| `entity_heuristic_w5` | Entity-based chunking, window size 5 | ❌ Not implemented | Medium |
| `entity_heuristic_w6` | Entity-based chunking, window size 6 | ❌ Not implemented | Medium |
| `Heading_sections_l2` | Heading-based sections, level 2 | ❌ Not implemented | Low |

**Current Implementation:** Sentence-based chunking only

### Dimension 2: Embedding Models (3 options)

| Model | Type | Dimension | Provider | Status | Cost |
|-------|------|-----------|----------|--------|------|
| `jina_v3` | Open Source | TBD (confirm during implementation) | Jina AI | ❌ Not implemented | Free/API |
| `gte_multilingual_base` | Open Source | TBD (confirm during implementation) | Alibaba | ❌ Not implemented | Free |
| `openai_text-embedding-3-large` | Commercial | 3072 | OpenAI | ❌ Not implemented | Paid |

**Current Implementation:** NVIDIA nv-embedqa-e5-v5 (1024 dim)

**Note:** Embedding dimensions must be confirmed during implementation. Different dimensions require separate vector DB collections.

### Dimension 3: Vector Databases (4 options)

| Database | Type | Status | Setup Complexity | Features |
|----------|------|--------|------------------|----------|
| `Qdrant` | Open Source | ❌ Not implemented | Medium | Fast, filtering, cloud |
| `PGVector` | Open Source | ❌ Not implemented | Medium | PostgreSQL extension |
| `Weaviate` | Open Source | ❌ Not implemented | Medium | GraphQL, hybrid search |
| `Milvus` | Open Source | ✅ Implemented | Low | Current system |

**Current Implementation:** Milvus only

### Dimension 4: Index Type (1 option)

| Index Type | Description | Status | Performance |
|------------|-------------|--------|-------------|
| `HNSW` | Hierarchical Navigable Small World | ❌ Not configured | High speed, high recall (expected) |

**Current Implementation:** IVF_FLAT

**Important:** HNSW is NOT currently implemented or configured. Milvus supports HNSW, but it must be:
1. Configured in collection creation
2. Tested for correctness
3. Benchmarked for performance
4. Compared against IVF_FLAT baseline

### Dimension 5: Retrieval Method (1 option)

| Method | Description | Status |
|--------|-------------|--------|
| `Cosine Similarity` | Vector cosine similarity | ✅ Implemented |

**Current Implementation:** Cosine similarity with COSINE metric

### Dimension 6: Reranking Models (3 options + baseline)

| Model | Type | Provider | Status | Cost | Purpose |
|-------|------|----------|--------|------|---------|
| `Amazon Rerank v1` | Commercial | AWS | ❌ Not implemented | Paid | Official benchmark |
| `Qwen3:4B Rerank` | Open Source | Alibaba | ❌ Not implemented | Free | Official benchmark |
| `bge-reranker-base` | Open Source | BAAI | ❌ Not implemented | Free | Official benchmark |
| `no_reranker` | N/A | N/A | ✅ Current state | Free | Engineering baseline |

**Current Implementation:** No reranking layer

**Benchmark Strategy:**
- **Official Matrix:** 144 combinations (3 rerankers as specified in meeting)
- **Engineering Baseline:** Add `no_reranker` for comparison (192 total combinations)
- **Minimum Requirement:** Run current baseline with `no_reranker` separately to establish baseline metrics

---

## Benchmark Data Schema

### Input Data Format

**Required Columns:**
```csv
id,pdf_name,paragraph
```

**Example:**
```csv
1,flight_manual.pdf,"Customer has Multi PNR and received Voucher email only for one PNR..."
2,flight_manual.pdf,"Open ticket is when we leave the PNR alive with a retention line..."
3,refund_policy.pdf,"Virtual interlining has a similar meaning to traditional interlining..."
```

**Current Data Issue:** Existing `qa_text_test.csv` uses format:
```csv
question|ground_truth|context
```

**Action Required:** Convert existing data or create new dataset

### Benchmark Configuration Schema (Proposed)

**Columns:**
```csv
benchmark_run_id,chunking_method,embedding_model,vector_database,index_type,retrieval_method,reranking_model,commercial_or_open_source
```

**Example (illustrative only, not actual data):**
```csv
baseline_001,sentence_based,nvidia_nv-embedqa-e5-v5,Milvus,IVF_FLAT,cosine_similarity,no_reranker,open_source
run_001,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,bge-reranker-base,open_source
run_002,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,Amazon_Rerank_v1,mixed
run_003,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,Qwen3_4B_Rerank,open_source
```

### Benchmark Results Schema (Proposed)

**Columns:**
```csv
benchmark_run_id,query_id,pdf_name,paragraph_id,chunking_method,embedding_model,vector_database,index_type,retrieval_method,reranking_model,commercial_or_open_source,retrieval_score,answer_quality_score,indexing_latency_ms,retrieval_latency_ms,reranking_latency_ms,total_latency_ms,notes
```

**Metrics Breakdown (to be implemented):**
- `retrieval_score`: Precision@k, Recall@k, MRR, NDCG (formulas to be confirmed)
- `answer_quality_score`: ROUGE-L, BERTScore, BLEURT, Cosine Similarity (partially exists)
- `*_latency_ms`: Time measurements in milliseconds (to be implemented)
- `notes`: Error messages, warnings, observations

---

## How Many Combinations

### Current Baseline (Existing System)
```
1 configuration:
- Chunking: sentence_based (existing)
- Embedding: NVIDIA nv-embedqa-e5-v5 (existing)
- Vector DB: Milvus (existing)
- Index: IVF_FLAT (existing)
- Retrieval: COSINE (existing)
- Reranking: no_reranker (existing - no reranking layer)
```

### Official Benchmark Matrix
```
Chunking (4) × Embedding (3) × VectorDB (4) × Index (1) × Retrieval (1) × Reranking (3)
= 4 × 3 × 4 × 1 × 1 × 3
= 144 total combinations
```

### Engineering Matrix (with no_reranker baseline)
```
Chunking (4) × Embedding (3) × VectorDB (4) × Index (1) × Retrieval (1) × Reranking (4)
= 4 × 3 × 4 × 1 × 1 × 4
= 192 total combinations (includes no_reranker for comparison)
```

### Breakdown by Category

**By Commercial vs Open Source:**
- Pure Open Source: 4 × 2 × 4 × 1 × 1 × 2 = 64 combinations
- Mixed (some commercial): 80 combinations
- Pure Commercial: 4 × 1 × 4 × 1 × 1 × 1 = 16 combinations

**By Vector Database:**
- Qdrant: 36 combinations (to be implemented)
- PGVector: 36 combinations (to be implemented)
- Weaviate: 36 combinations (to be implemented)
- Milvus: 36 combinations (1 baseline exists, 35 to be tested)

**By Embedding Model:**
- jina_v3: 48 combinations (to be implemented)
- gte_multilingual_base: 48 combinations (to be implemented)
- openai_text-embedding-3-large: 48 combinations (to be implemented)

**By Reranking Model:**
- Amazon Rerank v1: 48 combinations (to be implemented)
- Qwen3:4B Rerank: 48 combinations (to be implemented)
- bge-reranker-base: 48 combinations (to be implemented)
- no_reranker: 48 combinations (baseline for comparison)

### Phased Approach (Recommended)

**Phase 0: Current Baseline**
- 1 combination: Current system as-is
- Purpose: Establish baseline metrics
- Estimated time: ❓ To be determined after MVP

**Phase 1: MVP Benchmark**
- 1 combination: Current baseline with proper data format and metrics
- Purpose: Validate benchmark framework
- Estimated time: ❓ To be determined

**Phase 2: Chunking Variations**
- 4 combinations: Test all chunking methods with current stack
- Estimated time: ❓ To be determined after Phase 1

**Phase 3: Embedding Variations**
- 12 combinations: 4 chunking × 3 embeddings (current VectorDB)
- Estimated time: ❓ To be determined after Phase 2

**Phase 4: Vector Database Variations**
- 48 combinations: 4 chunking × 3 embeddings × 4 VectorDBs
- Estimated time: ❓ To be determined after Phase 3

**Phase 5: Reranking Variations**
- 144 combinations: Full official matrix
- Estimated time: ❓ To be determined after Phase 4

**Note:** All time estimates are placeholders. Actual execution time depends on:
- Dataset size (number of queries)
- Hardware resources
- API rate limits
- Implementation efficiency

---


## Implementation Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Orchestrator                        │
│  - Configuration matrix generation                               │
│  - Parallel execution management                                 │
│  - Result aggregation                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌──────▼──────────┐
│   Component    │   │   Benchmark     │   │    Results      │
│   Abstractions │   │   Runner        │   │    Storage      │
└────────────────┘   └─────────────────┘   └─────────────────┘
```

### Required Abstractions

#### 1. Chunking Strategy Interface (Proposed)
```python
# PROPOSED - Not yet implemented
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> List[Chunk]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

**Implementations Needed:**
- `EntityHeuristicChunker(window_size: int)` - To be implemented
- `HeadingSectionChunker(level: int)` - To be implemented
- `SentenceChunker()` - Wrap existing logic from `embedding_service.py`

#### 2. Embedding Provider Interface (Proposed)
```python
# PROPOSED - Not yet implemented
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

**Implementations Needed:**
- `JinaV3Provider()` - To be implemented
- `GTEMultilingualProvider()` - To be implemented
- `OpenAIEmbeddingProvider(model="text-embedding-3-large")` - To be implemented
- `NVIDIAProvider()` - Wrap existing implementation from `embedding_service.py`

#### 3. Vector Database Interface (Proposed)
```python
# PROPOSED - Not yet implemented
class VectorDatabase(ABC):
    @abstractmethod
    async def create_collection(self, name: str, dimension: int, index_type: str):
        pass
    
    @abstractmethod
    async def insert(self, collection: str, vectors: List[List[float]], 
                     metadata: List[dict]):
        pass
    
    @abstractmethod
    async def search(self, collection: str, query_vector: List[float], 
                     top_k: int) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str):
        pass
```

**Implementations Needed:**
- `QdrantDatabase()` - To be implemented
- `PGVectorDatabase()` - To be implemented
- `WeaviateDatabase()` - To be implemented
- `MilvusDatabase()` - Wrap existing implementation from `vector_service.py`

#### 4. Reranker Interface (Proposed)
```python
# PROPOSED - Not yet implemented
class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: List[str], 
                     top_k: int) -> List[Tuple[int, float]]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

**Implementations Needed:**
- `AmazonRerankV1()` - To be implemented
- `Qwen3RerankProvider()` - To be implemented
- `BGERerankProvider()` - To be implemented
- `NoReranker()` - To be implemented (pass-through for baseline)

---

## Code Files to Modify or Create

### New Files to Create (Proposed)

⚠️ **All files in this section are PROPOSED and not yet implemented.**

#### 1. `source/benchmark/` (New Package - To Be Created)

**`source/benchmark/__init__.py`** (Proposed)
- Package initialization

**`source/benchmark/config.py`** (Proposed)
- Benchmark configuration dataclasses
- Matrix generation logic
- Configuration validation

**`source/benchmark/orchestrator.py`** (Proposed)
- Main benchmark orchestrator
- Parallel execution management
- Progress tracking
- Error handling and retry logic

**`source/benchmark/runner.py`** (Proposed)
- Single benchmark run execution
- Pipeline: chunk → embed → index → search → rerank → evaluate
- Timing and metric collection

**`source/benchmark/results.py`** (Proposed)
- Result storage (CSV, JSON, database)
- Result aggregation and analysis
- Report generation

**`source/benchmark/metrics.py`** (Proposed)
- Evaluation metrics implementation
- Precision, Recall, MRR, NDCG
- ROUGE, BERTScore, BLEURT wrappers
- Latency tracking

#### 2. `source/chunking/` (New Package - To Be Created)

**`source/chunking/__init__.py`** (Proposed)
- Chunking strategy registry

**`source/chunking/base.py`** (Proposed)
- `ChunkingStrategy` abstract base class
- `Chunk` dataclass

**`source/chunking/entity_heuristic.py`** (Proposed)
- `EntityHeuristicChunker` implementation
- Window-based entity extraction
- Configurable window sizes (4, 5, 6)

**`source/chunking/heading_sections.py`** (Proposed)
- `HeadingSectionChunker` implementation
- Markdown/HTML heading detection
- Level-based section extraction

**`source/chunking/sentence.py`** (Proposed)
- `SentenceChunker` wrapper around existing logic
- Migrate from `embedding_service.py`

#### 3. `source/embeddings/` (New Package - To Be Created)

**`source/embeddings/__init__.py`** (Proposed)
- Embedding provider registry

**`source/embeddings/base.py`** (Proposed)
- `EmbeddingProvider` abstract base class

**`source/embeddings/jina.py`** (Proposed)
- `JinaV3Provider` implementation
- API client for Jina embeddings

**`source/embeddings/gte.py`** (Proposed)
- `GTEMultilingualProvider` implementation
- Hugging Face transformers integration

**`source/embeddings/openai.py`** (Proposed)
- `OpenAIEmbeddingProvider` implementation
- OpenAI API client

**`source/embeddings/nvidia.py`** (Proposed)
- Wrapper around existing NVIDIA provider
- Conform to new interface

#### 4. `source/vector_dbs/` (New Package)

**`source/vector_dbs/__init__.py`**
- Vector database registry

**`source/vector_dbs/base.py`**
- `VectorDatabase` abstract base class
- `SearchResult` dataclass

**`source/vector_dbs/qdrant.py`**
- `QdrantDatabase` implementation
- qdrant-client integration

**`source/vector_dbs/pgvector.py`**
- `PGVectorDatabase` implementation
- psycopg2/asyncpg integration

**`source/vector_dbs/weaviate.py`**
- `WeaviateDatabase` implementation
- weaviate-client integration

**`source/vector_dbs/milvus.py`**
- Wrapper around existing Milvus service
- Conform to new interface

#### 5. `source/reranking/` (New Package)

**`source/reranking/__init__.py`**
- Reranker registry

**`source/reranking/base.py`**
- `Reranker` abstract base class

**`source/reranking/amazon.py`**
- `AmazonRerankV1` implementation
- AWS Bedrock integration

**`source/reranking/qwen.py`**
- `Qwen3RerankProvider` implementation
- Model loading and inference

**`source/reranking/bge.py`**
- `BGERerankProvider` implementation
- Hugging Face transformers integration

**`source/reranking/no_rerank.py`**
- `NoReranker` pass-through implementation

#### 6. `scripts/` (New Directory)

**`scripts/run_benchmark.py`**
- CLI entry point for benchmarking
- Argument parsing
- Configuration loading
- Orchestrator invocation

**`scripts/generate_matrix.py`**
- Generate benchmark configuration matrix
- Export to CSV

**`scripts/analyze_results.py`**
- Result analysis and visualization
- Statistical comparisons
- Best configuration identification

**`scripts/prepare_data.py`**
- Convert existing test data to benchmark format
- Extract paragraphs from PDFs
- Generate query-paragraph pairs

### Files to Modify

#### 1. `source/services/embedding_service.py`
- **Changes:**
  - Extract chunking logic to `source/chunking/sentence.py`
  - Keep provider abstraction, add new providers
  - Maintain backward compatibility

#### 2. `source/services/vector_service.py`
- **Changes:**
  - Wrap in new `VectorDatabase` interface
  - Keep existing functionality
  - Add HNSW index configuration

#### 3. `config/QA_Text/config.yaml`
- **Changes:**
  - Add benchmark configuration section
  - Add new provider credentials
  - Add chunking method configurations

#### 4. `requirements.txt`
- **Changes:**
  - Add new dependencies:
    - `qdrant-client`
    - `psycopg2-binary` or `asyncpg`
    - `weaviate-client`
    - `jina` or `jinaai`
    - `sentence-transformers` (already present)
    - `openai` (already present)
    - `boto3` (for Amazon Rerank)

#### 5. `main.py`
- **Changes:**
  - Add benchmark router (optional, for API-based benchmarking)
  - Keep existing functionality

---

## Implementation Tasks (Ordered)

### Phase 0: Preparation (Week 1)

- [ ] **Task 0.1:** Review and approve NOTES.md and BENCHMARK_PLAN.md
- [ ] **Task 0.2:** Set up development branch (`feature/benchmarking`)
- [ ] **Task 0.3:** Confirm data availability and format
- [ ] **Task 0.4:** Set up additional vector database instances (Docker)
- [ ] **Task 0.5:** Obtain API keys for commercial services
- [ ] **Task 0.6:** Create test dataset in correct format (`id|pdf_name|paragraph`)

### Phase 1: Core Abstractions (Week 2)

- [ ] **Task 1.1:** Create `source/benchmark/` package structure
- [ ] **Task 1.2:** Implement `ChunkingStrategy` base class and registry
- [ ] **Task 1.3:** Implement `EmbeddingProvider` base class and registry
- [ ] **Task 1.4:** Implement `VectorDatabase` base class and registry
- [ ] **Task 1.5:** Implement `Reranker` base class and registry
- [ ] **Task 1.6:** Create benchmark configuration dataclasses
- [ ] **Task 1.7:** Write unit tests for abstractions

### Phase 2: Chunking Implementations (Week 3)

- [ ] **Task 2.1:** Implement `SentenceChunker` (migrate existing logic)
- [ ] **Task 2.2:** Implement `EntityHeuristicChunker` (window_size=4)
- [ ] **Task 2.3:** Implement `EntityHeuristicChunker` (window_size=5)
- [ ] **Task 2.4:** Implement `EntityHeuristicChunker` (window_size=6)
- [ ] **Task 2.5:** Implement `HeadingSectionChunker` (level=2)
- [ ] **Task 2.6:** Write unit tests for each chunker
- [ ] **Task 2.7:** Validate chunking output quality manually

### Phase 3: Embedding Implementations (Week 4)

- [ ] **Task 3.1:** Wrap existing NVIDIA provider in new interface
- [ ] **Task 3.2:** Implement `JinaV3Provider`
- [ ] **Task 3.3:** Implement `GTEMultilingualProvider`
- [ ] **Task 3.4:** Implement `OpenAIEmbeddingProvider`
- [ ] **Task 3.5:** Write unit tests for each provider
- [ ] **Task 3.6:** Test dimension compatibility with vector DBs
- [ ] **Task 3.7:** Benchmark embedding generation speed

### Phase 4: Vector Database Implementations (Week 5-6)

- [ ] **Task 4.1:** Wrap existing Milvus service in new interface
- [ ] **Task 4.2:** Configure Milvus with HNSW index
- [ ] **Task 4.3:** Implement `QdrantDatabase`
- [ ] **Task 4.4:** Implement `PGVectorDatabase`
- [ ] **Task 4.5:** Implement `WeaviateDatabase`
- [ ] **Task 4.6:** Write integration tests for each database
- [ ] **Task 4.7:** Test CRUD operations and search quality
- [ ] **Task 4.8:** Benchmark indexing and search speed

### Phase 5: Reranking Implementations (Week 7)

- [ ] **Task 5.1:** Implement `NoReranker` (pass-through)
- [ ] **Task 5.2:** Implement `AmazonRerankV1`
- [ ] **Task 5.3:** Implement `Qwen3RerankProvider`
- [ ] **Task 5.4:** Implement `BGERerankProvider`
- [ ] **Task 5.5:** Write unit tests for each reranker
- [ ] **Task 5.6:** Test reranking quality manually
- [ ] **Task 5.7:** Benchmark reranking latency

### Phase 6: Benchmark Framework (Week 8)

- [ ] **Task 6.1:** Implement benchmark configuration matrix generator
- [ ] **Task 6.2:** Implement single benchmark runner
- [ ] **Task 6.3:** Implement benchmark orchestrator with parallel execution
- [ ] **Task 6.4:** Implement metrics collection (retrieval + answer quality)
- [ ] **Task 6.5:** Implement latency tracking
- [ ] **Task 6.6:** Implement result storage (CSV export)
- [ ] **Task 6.7:** Implement progress tracking and logging
- [ ] **Task 6.8:** Implement error handling and retry logic

### Phase 7: CLI and Scripts (Week 9)

- [ ] **Task 7.1:** Create `scripts/run_benchmark.py` CLI
- [ ] **Task 7.2:** Create `scripts/generate_matrix.py`
- [ ] **Task 7.3:** Create `scripts/analyze_results.py`
- [ ] **Task 7.4:** Create `scripts/prepare_data.py`
- [ ] **Task 7.5:** Write documentation for each script
- [ ] **Task 7.6:** Test CLI with small dataset

### Phase 8: Testing and Validation (Week 10)

- [ ] **Task 8.1:** Run pilot benchmark (10 combinations)
- [ ] **Task 8.2:** Validate result format and completeness
- [ ] **Task 8.3:** Fix bugs and edge cases
- [ ] **Task 8.4:** Optimize performance bottlenecks
- [ ] **Task 8.5:** Run full benchmark (144 combinations)
- [ ] **Task 8.6:** Generate analysis report
- [ ] **Task 8.7:** Identify best configurations

### Phase 9: Documentation and Handoff (Week 11)

- [ ] **Task 9.1:** Write comprehensive README for benchmark framework
- [ ] **Task 9.2:** Document each component and interface
- [ ] **Task 9.3:** Create usage examples and tutorials
- [ ] **Task 9.4:** Document findings and recommendations
- [ ] **Task 9.5:** Prepare presentation for stakeholders
- [ ] **Task 9.6:** Code review and merge to main

---

## Testing and Validation Steps

### Unit Testing

**For Each Component:**
1. Test initialization with valid/invalid parameters
2. Test core functionality with sample data
3. Test error handling and edge cases
4. Test resource cleanup (connections, files)

**Coverage Target:** >80% for new code

### Integration Testing

**For Each Pipeline Stage:**
1. Test chunking → embedding integration
2. Test embedding → vector DB integration
3. Test vector DB → reranking integration
4. Test end-to-end pipeline

**Test Data:** Use small subset (10-20 documents)

### Benchmark Validation

**Pilot Run (10 combinations):**
1. Select diverse configurations
2. Run on small dataset (50 queries)
3. Validate output format
4. Check for errors and warnings
5. Measure execution time
6. Review sample results manually

**Full Run (144 combinations):**
1. Run on full dataset (500+ queries)
2. Monitor resource usage (CPU, memory, disk)
3. Track progress and ETA
4. Handle failures gracefully
5. Generate comprehensive report

### Quality Checks

**Data Quality:**
- [ ] All input data has required columns
- [ ] No missing or malformed values
- [ ] PDF sources are accessible
- [ ] Paragraphs are properly extracted

**Result Quality:**
- [ ] All metrics are calculated correctly
- [ ] Latency measurements are reasonable
- [ ] No NaN or infinite values
- [ ] Results are reproducible

**System Quality:**
- [ ] No memory leaks
- [ ] Proper resource cleanup
- [ ] Graceful error handling
- [ ] Clear logging and progress tracking

---


## How to Run One Benchmark

### Prerequisites
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Ensure vector databases are running
docker ps | grep -E "milvus|qdrant|weaviate|postgres"

# Verify API keys are set
echo $NVIDIA_API_KEY
echo $OPENAI_API_KEY
echo $AWS_ACCESS_KEY_ID
```

### Single Configuration Run

**Step 1: Prepare Configuration**
```python
# config.json
{
  "benchmark_run_id": "test_001",
  "chunking_method": "entity_heuristic_w4",
  "embedding_model": "jina_v3",
  "vector_database": "Qdrant",
  "index_type": "HNSW",
  "retrieval_method": "cosine_similarity",
  "reranking_model": "bge-reranker-base",
  "data_path": "data/benchmark_input.csv",
  "output_path": "data/benchmark_results.csv"
}
```

**Step 2: Run Benchmark**
```bash
# Using CLI script
python scripts/run_benchmark.py \
  --config config.json \
  --verbose

# Or using Python API
python -c "
from source.benchmark.orchestrator import BenchmarkOrchestrator
from source.benchmark.config import BenchmarkConfig

config = BenchmarkConfig.from_file('config.json')
orchestrator = BenchmarkOrchestrator()
results = orchestrator.run_single(config)
print(results.summary())
"
```

**Step 3: Review Results**
```bash
# Check output CSV
head -n 5 data/benchmark_results.csv

# View summary statistics
python scripts/analyze_results.py \
  --input data/benchmark_results.csv \
  --summary
```

### Expected Output

**Console Output:**
```
[2026-05-13 10:00:00] INFO: Starting benchmark run_001
[2026-05-13 10:00:01] INFO: Loading data from data/benchmark_input.csv
[2026-05-13 10:00:02] INFO: Loaded 500 queries
[2026-05-13 10:00:03] INFO: Initializing chunker: entity_heuristic_w4
[2026-05-13 10:00:04] INFO: Initializing embedding provider: jina_v3
[2026-05-13 10:00:05] INFO: Initializing vector database: Qdrant
[2026-05-13 10:00:06] INFO: Creating collection: benchmark_run_001
[2026-05-13 10:00:10] INFO: Chunking documents... (0/100)
[2026-05-13 10:01:00] INFO: Chunking complete. 2,450 chunks generated.
[2026-05-13 10:01:01] INFO: Generating embeddings... (0/2450)
[2026-05-13 10:05:00] INFO: Embeddings complete. Dimension: 1024
[2026-05-13 10:05:01] INFO: Indexing vectors... (0/2450)
[2026-05-13 10:06:00] INFO: Indexing complete. Index type: HNSW
[2026-05-13 10:06:01] INFO: Running queries... (0/500)
[2026-05-13 10:15:00] INFO: Queries complete.
[2026-05-13 10:15:01] INFO: Calculating metrics...
[2026-05-13 10:16:00] INFO: Benchmark complete.
[2026-05-13 10:16:01] INFO: Results saved to data/benchmark_results.csv

Summary:
  Avg Retrieval Score: 0.78
  Avg Answer Quality: 0.82
  Avg Indexing Latency: 24.5 ms
  Avg Retrieval Latency: 45.2 ms
  Avg Reranking Latency: 12.3 ms
  Avg Total Latency: 81.0 ms
```

**CSV Output (sample rows):**
```csv
benchmark_run_id,query_id,pdf_name,paragraph_id,chunking_method,embedding_model,vector_database,index_type,retrieval_method,reranking_model,commercial_or_open_source,retrieval_score,answer_quality_score,indexing_latency_ms,retrieval_latency_ms,reranking_latency_ms,total_latency_ms,notes
test_001,1,flight_manual.pdf,42,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,bge-reranker-base,open_source,0.85,0.79,23.4,42.1,11.2,76.7,
test_001,2,flight_manual.pdf,108,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,bge-reranker-base,open_source,0.72,0.81,24.1,48.3,13.4,85.8,
test_001,3,refund_policy.pdf,15,entity_heuristic_w4,jina_v3,Qdrant,HNSW,cosine_similarity,bge-reranker-base,open_source,0.91,0.88,25.2,41.5,10.9,77.6,
```

---

## How to Run All Benchmarks

### Full Matrix Execution

**Step 1: Generate Configuration Matrix**
```bash
python scripts/generate_matrix.py \
  --output configs/benchmark_matrix.csv \
  --include-baseline

# Output: 144 configuration files in configs/
```

**Step 2: Run All Benchmarks (Parallel)**
```bash
python scripts/run_benchmark.py \
  --matrix configs/benchmark_matrix.csv \
  --output-dir results/ \
  --workers 8 \
  --resume-on-failure

# Estimated time: 18-36 hours with 8 workers
```

**Step 3: Monitor Progress**
```bash
# In another terminal
tail -f logs/benchmark.log

# Or use progress dashboard
python scripts/monitor_progress.py \
  --log-file logs/benchmark.log
```

**Step 4: Aggregate Results**
```bash
python scripts/analyze_results.py \
  --input-dir results/ \
  --output results/aggregated_results.csv \
  --generate-report results/benchmark_report.html
```

### Phased Execution (Recommended)

**Phase 1: Baseline**
```bash
python scripts/run_benchmark.py \
  --config configs/baseline.json \
  --output results/phase1_baseline.csv
```

**Phase 2: Chunking Variations**
```bash
python scripts/run_benchmark.py \
  --matrix configs/phase2_chunking.csv \
  --output-dir results/phase2/ \
  --workers 4
```

**Phase 3: Embedding Variations**
```bash
python scripts/run_benchmark.py \
  --matrix configs/phase3_embeddings.csv \
  --output-dir results/phase3/ \
  --workers 4
```

**Phase 4: Vector DB Variations**
```bash
python scripts/run_benchmark.py \
  --matrix configs/phase4_vectordb.csv \
  --output-dir results/phase4/ \
  --workers 4
```

**Phase 5: Reranking Variations**
```bash
python scripts/run_benchmark.py \
  --matrix configs/phase5_reranking.csv \
  --output-dir results/phase5/ \
  --workers 8
```

**Phase 6: Aggregate All Phases**
```bash
python scripts/analyze_results.py \
  --input-dirs results/phase*/ \
  --output results/final_results.csv \
  --generate-report results/final_report.html
```

---

## Expected Outputs

### 1. Raw Results CSV

**File:** `results/benchmark_results_YYYYMMDD_HHMMSS.csv`

**Columns:**
- `benchmark_run_id`: Unique identifier for configuration
- `query_id`: Query identifier from input data
- `pdf_name`: Source PDF document
- `paragraph_id`: Paragraph identifier
- `chunking_method`: Chunking strategy used
- `embedding_model`: Embedding model used
- `vector_database`: Vector database used
- `index_type`: Index type (HNSW)
- `retrieval_method`: Retrieval method (cosine_similarity)
- `reranking_model`: Reranking model used
- `commercial_or_open_source`: Cost category
- `retrieval_score`: Retrieval quality (0-1)
- `answer_quality_score`: Answer quality (0-1)
- `indexing_latency_ms`: Time to index (milliseconds)
- `retrieval_latency_ms`: Time to retrieve (milliseconds)
- `reranking_latency_ms`: Time to rerank (milliseconds)
- `total_latency_ms`: End-to-end latency (milliseconds)
- `notes`: Error messages or warnings

**Size:** ~72,000 rows (500 queries × 144 configs)

### 2. Aggregated Results CSV

**File:** `results/aggregated_results.csv`

**Columns:**
- `benchmark_run_id`: Configuration identifier
- `chunking_method`: Chunking strategy
- `embedding_model`: Embedding model
- `vector_database`: Vector database
- `index_type`: Index type
- `retrieval_method`: Retrieval method
- `reranking_model`: Reranking model
- `commercial_or_open_source`: Cost category
- `avg_retrieval_score`: Mean retrieval score
- `std_retrieval_score`: Std dev retrieval score
- `avg_answer_quality`: Mean answer quality
- `std_answer_quality`: Std dev answer quality
- `avg_indexing_latency_ms`: Mean indexing time
- `avg_retrieval_latency_ms`: Mean retrieval time
- `avg_reranking_latency_ms`: Mean reranking time
- `avg_total_latency_ms`: Mean total latency
- `p50_latency_ms`: Median latency
- `p95_latency_ms`: 95th percentile latency
- `p99_latency_ms`: 99th percentile latency
- `error_count`: Number of failed queries
- `success_rate`: Percentage of successful queries

**Size:** 144 rows (one per configuration)

### 3. Summary Statistics

**File:** `results/summary_statistics.json`

**Content:**
```json
{
  "total_configurations": 144,
  "total_queries": 500,
  "total_runs": 72000,
  "successful_runs": 71523,
  "failed_runs": 477,
  "success_rate": 0.993,
  "execution_time_hours": 24.5,
  "best_configuration": {
    "benchmark_run_id": "run_087",
    "chunking_method": "entity_heuristic_w5",
    "embedding_model": "openai_text-embedding-3-large",
    "vector_database": "Qdrant",
    "reranking_model": "Amazon_Rerank_v1",
    "avg_retrieval_score": 0.89,
    "avg_answer_quality": 0.91,
    "avg_total_latency_ms": 78.3
  },
  "best_open_source": {
    "benchmark_run_id": "run_042",
    "chunking_method": "entity_heuristic_w6",
    "embedding_model": "jina_v3",
    "vector_database": "Milvus",
    "reranking_model": "bge-reranker-base",
    "avg_retrieval_score": 0.84,
    "avg_answer_quality": 0.86,
    "avg_total_latency_ms": 65.2
  },
  "fastest_configuration": {
    "benchmark_run_id": "run_015",
    "avg_total_latency_ms": 42.1
  },
  "most_accurate_configuration": {
    "benchmark_run_id": "run_087",
    "avg_answer_quality": 0.91
  }
}
```

### 4. HTML Report

**File:** `results/benchmark_report.html`

**Sections:**
1. **Executive Summary**
   - Best overall configuration
   - Best open-source configuration
   - Key findings and recommendations

2. **Performance Comparison**
   - Bar charts: Retrieval score by configuration
   - Bar charts: Answer quality by configuration
   - Bar charts: Latency by configuration

3. **Component Analysis**
   - Chunking method comparison
   - Embedding model comparison
   - Vector database comparison
   - Reranking model comparison

4. **Cost Analysis**
   - Open source vs commercial comparison
   - API cost estimates
   - Compute cost estimates

5. **Detailed Results**
   - Sortable table with all configurations
   - Filtering by component
   - Export to CSV

6. **Recommendations**
   - Production deployment suggestions
   - Trade-off analysis (accuracy vs latency vs cost)
   - Next steps

### 5. Visualization Outputs

**Files:**
- `results/charts/retrieval_score_by_chunking.png`
- `results/charts/retrieval_score_by_embedding.png`
- `results/charts/retrieval_score_by_vectordb.png`
- `results/charts/retrieval_score_by_reranking.png`
- `results/charts/latency_distribution.png`
- `results/charts/accuracy_vs_latency_scatter.png`
- `results/charts/cost_vs_accuracy_scatter.png`

---

## Suggested Output File Format

### Primary Format: CSV

**Rationale:**
- Easy to import into Excel, Google Sheets, Pandas
- Human-readable
- Version control friendly
- Widely supported

**Delimiter:** Comma (`,`)
**Encoding:** UTF-8
**Line Endings:** LF (`\n`)

### Secondary Format: JSON Lines (JSONL)

**File:** `results/benchmark_results.jsonl`

**Rationale:**
- Preserves nested structures
- Easy to stream and process
- Good for programmatic access

**Example:**
```json
{"benchmark_run_id": "run_001", "query_id": 1, "retrieval_score": 0.85, "answer_quality_score": 0.79, ...}
{"benchmark_run_id": "run_001", "query_id": 2, "retrieval_score": 0.72, "answer_quality_score": 0.81, ...}
```

### Tertiary Format: Parquet

**File:** `results/benchmark_results.parquet`

**Rationale:**
- Columnar storage (efficient for analytics)
- Compression (smaller file size)
- Fast read/write with Pandas/Polars

**Usage:**
```python
import pandas as pd
df = pd.read_parquet('results/benchmark_results.parquet')
```

---

## Assumptions and Open Questions

### Assumptions

1. **Data Availability:**
   - ✅ PDF source files are available
   - ✅ Paragraphs can be extracted reliably
   - ✅ Ground truth answers exist for evaluation

2. **Infrastructure:**
   - ✅ Local machine has sufficient resources (16GB+ RAM, 8+ CPU cores)
   - ✅ Vector databases can run in Docker
   - ✅ API rate limits are acceptable

3. **API Access:**
   - ⚠️ NVIDIA API key is available (confirmed in config)
   - ❓ OpenAI API key is available
   - ❓ AWS credentials for Amazon Rerank are available
   - ❓ Jina API key is available (or can use local models)

4. **Evaluation:**
   - ✅ ROUGE/BERTScore/BLEURT are sufficient metrics
   - ✅ Cosine similarity is a good proxy for answer quality
   - ⚠️ Ground truth answers are high quality

5. **Performance:**
   - ✅ 1 hour per configuration is acceptable
   - ✅ Parallel execution with 8 workers is feasible
   - ✅ 24-36 hours total execution time is acceptable

### Open Questions

1. **Data:**
   - ❓ Do we have the original PDFs for all test cases in `qa_text_test.csv`?
   - ❓ Should we use existing test data or create new benchmark dataset?
   - ❓ How many queries should we test per configuration? (50? 500? 5000?)
   - ❓ Do we need to balance queries across different PDF sources?

2. **Scope:**
   - ❓ Should we test ALL 144 combinations or prioritize certain configs?
   - ❓ Should we include ablation studies (e.g., no reranking)?
   - ❓ Should we test different top_k values (5, 10, 20)?
   - ❓ Should we test different similarity thresholds?

3. **Infrastructure:**
   - ❓ Can we use cloud resources (AWS, GCP, Azure) for faster execution?
   - ❓ Do we have access to GPU instances for embedding generation?
   - ❓ Should we set up dedicated vector DB instances or use Docker?

4. **Evaluation:**
   - ❓ Are ROUGE/BERTScore/BLEURT sufficient or do we need domain-specific metrics?
   - ❓ Should we include human evaluation for a subset of results?
   - ❓ Should we measure cost (API calls, compute time) in addition to quality?

5. **Implementation:**
   - ❓ Should we implement all components from scratch or use existing libraries?
   - ❓ Should we prioritize speed or code quality in the first iteration?
   - ❓ Should we build a web UI for monitoring or stick with CLI?

6. **Deliverables:**
   - ❓ What format should the final report be in? (PDF, HTML, Jupyter Notebook?)
   - ❓ Should we prepare a presentation for stakeholders?
   - ❓ Should we document the code for future maintenance?

7. **Timeline:**
   - ❓ What is the deadline for completing the benchmark?
   - ❓ Should we deliver results incrementally (phase by phase)?
   - ❓ Is there a budget constraint for API costs?

---

## Next Steps and Recommendations

### Immediate Actions (This Week)

1. **Review and Approve Documents**
   - [ ] Review NOTES.md with team
   - [ ] Review BENCHMARK_PLAN.md with team
   - [ ] Get sign-off from stakeholders

2. **Clarify Open Questions**
   - [ ] Schedule meeting to discuss open questions
   - [ ] Document decisions in this plan
   - [ ] Update timeline based on decisions

3. **Prepare Data**
   - [ ] Verify PDF availability
   - [ ] Convert existing test data to benchmark format
   - [ ] Create sample dataset (50 queries) for pilot

4. **Set Up Infrastructure**
   - [ ] Install Docker and pull vector DB images
   - [ ] Set up Qdrant, Weaviate, PGVector containers
   - [ ] Test connectivity to all vector DBs

5. **Obtain API Keys**
   - [ ] Request OpenAI API key
   - [ ] Request AWS credentials for Amazon Rerank
   - [ ] Request Jina API key (or plan to use local models)

### Short-Term Actions (Next 2 Weeks)

1. **Implement Core Abstractions**
   - [ ] Create package structure
   - [ ] Implement base classes
   - [ ] Write unit tests

2. **Implement First Components**
   - [ ] Wrap existing Milvus service
   - [ ] Wrap existing NVIDIA embedding provider
   - [ ] Implement sentence chunker wrapper

3. **Run Pilot Benchmark**
   - [ ] Test with 1-2 configurations
   - [ ] Validate output format
   - [ ] Identify and fix issues

### Medium-Term Actions (Next 4-6 Weeks)

1. **Implement All Components**
   - [ ] Complete all chunking methods
   - [ ] Complete all embedding providers
   - [ ] Complete all vector databases
   - [ ] Complete all reranking models

2. **Build Benchmark Framework**
   - [ ] Implement orchestrator
   - [ ] Implement parallel execution
   - [ ] Implement result storage

3. **Run Phased Benchmarks**
   - [ ] Phase 1: Baseline
   - [ ] Phase 2: Chunking variations
   - [ ] Phase 3: Embedding variations
   - [ ] Phase 4: Vector DB variations
   - [ ] Phase 5: Reranking variations

### Long-Term Actions (Next 8-12 Weeks)

1. **Run Full Benchmark**
   - [ ] Execute all 144 configurations
   - [ ] Monitor and troubleshoot
   - [ ] Aggregate results

2. **Analyze and Report**
   - [ ] Generate summary statistics
   - [ ] Create visualizations
   - [ ] Write final report
   - [ ] Prepare presentation

3. **Deploy Best Configuration**
   - [ ] Identify optimal configuration
   - [ ] Update production system
   - [ ] Monitor performance

---

## Success Criteria

### Technical Success

- ✅ All 144 configurations execute successfully
- ✅ Results are reproducible
- ✅ No data loss or corruption
- ✅ Execution completes within 48 hours
- ✅ All metrics are calculated correctly

### Quality Success

- ✅ Identify configuration with >10% improvement over baseline
- ✅ Identify best open-source configuration
- ✅ Identify fastest configuration with acceptable quality
- ✅ Statistical significance of differences (p < 0.05)

### Deliverable Success

- ✅ Complete CSV results file
- ✅ Comprehensive HTML report
- ✅ Clear recommendations for production
- ✅ Well-documented code
- ✅ Stakeholder presentation

---

**End of Benchmark Plan**
