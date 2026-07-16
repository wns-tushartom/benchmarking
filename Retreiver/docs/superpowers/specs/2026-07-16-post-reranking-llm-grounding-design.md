# Post-reranking LLM and grounding design

**Date:** 2026-07-16
**Status:** Approved design for a future implementation; not active in the meeting build

## 1. Purpose

Add an optional answer-generation layer after reranking without coupling the benchmark pipeline to one model vendor. The layer must generate answers only from retrieved evidence, preserve source citations, record provider and cost evidence, and produce artifacts that a separate grounding evaluator can audit.

The current meeting build remains a retrieval and reranking benchmark. It must not show generated-answer or grounding scores until this contract is implemented and verified.

## 2. Product boundary

The existing six-stage benchmark stays intact:

1. Extraction
2. Embedding
3. Vector-store ingestion
4. Retrieval
5. Reranking
6. Evaluation

Generation is an **optional extension** between Reranking and answer-level Evaluation. It does not replace retrieval evaluation and it does not change the official 180-combination matrix. A retrieval run can complete without generation.

Grounding audit stays disabled until generated-answer artifacts conform to this specification.

## 3. Recommended architecture

Use a provider-neutral generation service with four adapters:

- `OpenAIAdapter`
- `AnthropicAdapter`
- `GeminiAdapter`
- `OpenAICompatibleAdapter` for local/open-source servers such as vLLM, Ollama-compatible gateways, or llama.cpp-compatible endpoints

Provider SDK objects must not enter pipeline or artifact schemas. The orchestrator calls one internal interface and receives one normalized result.

```python
class GenerationAdapter(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

Provider configuration belongs in an allowlisted server-side catalog. The browser sends an opaque provider/model ID. API keys, endpoint URLs, and private model paths stay server-side.

## 4. Input contract

`GenerationRequest` contains:

- `run_id`
- `combination_id`
- `query_id`
- `query`
- ordered reranked evidence hits
- immutable evidence references for each hit:
  - project/source ID
  - chunk ID
  - source filename label
  - page number when available
  - retrieval rank
  - reranker score when available
  - exact evidence text used in the prompt
- provider catalog ID
- model catalog ID
- deterministic prompt-template version
- temperature, default `0`
- maximum answer tokens
- evidence-token budget

Evidence selection must be deterministic. The stage takes only the configured reranked-output K, then trims evidence to the token budget in rank order. It must never silently retrieve additional context.

## 5. Prompt and output contract

The system prompt must require the model to:

- answer only from supplied evidence
- say that the evidence is insufficient when support is absent
- attach citations using stable evidence IDs such as `[E1]`, `[E2]`
- avoid citing sources not supplied in the request
- return structured JSON when the provider supports it

Normalized `GenerationResult` contains:

- `status`: `completed`, `insufficient_evidence`, or `failed`
- `answer`
- `citations`: evidence IDs referenced by the answer
- `provider_id`
- `model_id`
- prompt-template version
- input/output token counts when reported
- token-count completeness flag
- latency
- provider request ID when available
- finish reason
- normalized error code and safe message on failure

Malformed provider output must fail that query row. It must not be converted into a plausible answer.

## 6. Citation integrity

Citation validation runs locally before any LLM-based judge:

1. Every cited ID must exist in the supplied evidence set.
2. Every citation stored in artifacts must resolve to an exact chunk and source reference.
3. Uncited answers are marked `citation_missing`.
4. Unknown citations are marked `citation_invalid`.
5. An `insufficient_evidence` response is valid without citations only when it makes no factual claim from the corpus.

Citation validity is a structural metric. It must not be presented as semantic grounding.

## 7. Grounding evaluation

Grounding evaluation is a separate module and artifact family. It must not reuse the answer-generating model as the only judge.

Evaluation layers:

1. **Deterministic checks**
   - citation validity
   - citation coverage
   - answer presence/refusal state
   - exact source/chunk resolution
2. **Reference evaluation**, only when answer-level ground truth exists
   - answer correctness/relevance under a versioned evaluator
3. **Claim-support evaluation**, optional
   - provider-neutral judge adapter
   - judge model recorded independently from generator model
   - claim-level supported/unsupported labels with cited evidence spans

If ground truth is absent, the UI may show answer, citations, latency, token usage, and deterministic citation integrity. It must not fabricate correctness, groundedness, or winner scores.

## 8. Artifact layout

Each project run writes immutable, run-owned artifacts:

```text
runs/<run_id>/
  generation/
    requests.jsonl
    answers.jsonl
    summary.json
  grounding/
    citation_checks.jsonl
    claim_support.jsonl        # only when judge executed
    summary.json
```

`answers.jsonl` stores normalized results, evidence references, provider/model IDs, prompt version, latency, and token usage. Secrets and provider endpoint URLs are forbidden.

The run manifest records:

- generation enabled/disabled
- provider/model catalog IDs
- prompt version
- generated, refused, and failed row counts
- grounding mode actually executed
- artifact hashes

Official retrieval/reranking artifacts remain unchanged.

## 9. Failure and retry rules

- Retry only documented transient provider errors: timeout, rate limit, and 5xx.
- Use bounded exponential backoff with a maximum attempt count in configuration.
- Do not retry invalid requests, authentication failures, policy refusals, or malformed structured output indefinitely.
- Failure is isolated per query.
- A run may be `partial` when some generation rows fail; retrieval evidence remains valid.
- Resumption uses request fingerprints and never overwrites a completed row with a different model/prompt configuration.

## 10. Security and privacy

- Credentials come only from server environment/configuration.
- Browser payloads use catalog IDs, never arbitrary provider URLs or model paths.
- Prompt artifacts must be treated as private because they contain corpus text.
- Provider eligibility can be restricted per dataset classification.
- Local/open-source mode is the safe path for corpora that cannot leave the VM.
- Logs redact API keys, authorization headers, and raw provider responses that may contain sensitive metadata.

## 11. UI activation criteria

Do not enable Generation or Grounding audit until all criteria pass:

- at least one provider adapter passes contract tests
- citation IDs round-trip to source evidence
- provider errors fail closed
- evidence-only mode suppresses correctness/grounding scores
- generated artifacts are readable by the dashboard without private-path leakage
- desktop and mobile visual QA pass
- a real end-to-end run produces answer, citation, usage, latency, and grounding artifacts

When activated, the UI should show provider/model selection only in a separate optional Generation section. The six-stage retrieval benchmark remains the default demo path.

## 12. Testing strategy

- Adapter contract tests shared by every provider.
- Golden tests for prompt construction and evidence-budget trimming.
- Citation validation tests for valid, missing, duplicate, and unknown citations.
- Ground-truth-none tests proving scored metrics are absent.
- Cross-project isolation tests for all generation and grounding artifacts.
- Resume/fingerprint tests preventing stale answer reuse.
- Provider failure tests for timeout, rate limit, authentication, malformed output, and partial runs.
- Real smoke tests per enabled provider, gated by credentials and excluded from ordinary unit runs.

## 13. Explicit non-goals for the meeting build

- No live answer generation.
- No enabled Grounding audit tab or buttons.
- No answer-level comparison mixed into retrieval winner scores.
- No browser-supplied API keys or arbitrary model endpoints.
- No claim that the existing lexical preview or retrieval evidence is generated-answer grounding.
