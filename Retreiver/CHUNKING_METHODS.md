# Chunking Methods Documentation

**Repo path:** `General_Components/QA_Text/Retreiver`  
**Input schema:** `id`, `pdf_name`, `paragraph`  
**Current output workbook:** `data/chunking_methods_output.xlsx`

## Verification status

Uploaded workbook verified successfully:

- Sheets found:
  - `original_input`
  - `entity_heuristic_w6`
  - `entity_heuristic_w5`
  - `entity_heuristic_w4`
  - `Heading_sections_l2`
- Every sheet has exact columns: `id`, `pdf_name`, `paragraph`
- No null `pdf_name`
- No null `paragraph`
- No duplicate IDs within sheets
- IDs are sequential within each sheet

### Row counts

- `original_input`: 61 rows
- `entity_heuristic_w6`: 214 chunks
- `entity_heuristic_w5`: 222 chunks
- `entity_heuristic_w4`: 227 chunks
- `Heading_sections_l2`: 127 chunks

## Important caveat

The current implementations of `entity_heuristic_w6`, `entity_heuristic_w5`, `entity_heuristic_w4`, and `Heading_sections_l2` should be treated as **heuristic approximations pending confirmation of official WNS definitions**.

Do not describe them as official WNS chunking methods unless the team confirms the exact intended algorithm.

---

## 1. `entity_heuristic_w6`

### Plain-English definition
Entity-aware chunking that tries to keep important business entities, numbers, acronyms, process names, and nearby context together. Uses a larger context window than `w5` and `w4`.

### Current implementation behavior
- Detects entity-like terms such as:
  - capitalized words
  - acronyms
  - numbers
  - mixed-case/domain terms
- Groups surrounding text around those entity anchors.
- Uses a window size of 6 to preserve wider context.

### Parameters
- `window_size = 6`
- Minimum chunk length: 50 characters
- Output columns: `id`, `pdf_name`, `paragraph`

### Observed output
- 214 chunks
- Average chunk length: ~188 characters

### Known limitations
- Entity detection is regex/heuristic-based, not true NER unless upgraded.
- Can preserve headers, exported metadata, and noisy PDF extraction text.
- May create overlapping chunks, increasing total rows and embedding cost.

---

## 2. `entity_heuristic_w5`

### Plain-English definition
Entity-aware chunking with medium context. Similar to `w6`, but uses a slightly smaller window.

### Current implementation behavior
- Detects entity-like terms.
- Groups nearby text around entity anchors.
- Uses window size 5.

### Parameters
- `window_size = 5`
- Minimum chunk length: 50 characters
- Output columns: `id`, `pdf_name`, `paragraph`

### Observed output
- 222 chunks
- Average chunk length: ~181 characters

### Known limitations
- Same as `w6`.
- May produce more chunks than `w6`, increasing embedding/vector DB rows.

---

## 3. `entity_heuristic_w4`

### Plain-English definition
Entity-aware chunking with tighter context. This produces smaller chunks and may improve precision, but can lose surrounding context.

### Current implementation behavior
- Detects entity-like terms.
- Groups nearby text around entity anchors.
- Uses window size 4.

### Parameters
- `window_size = 4`
- Minimum chunk length: 50 characters
- Output columns: `id`, `pdf_name`, `paragraph`

### Observed output
- 227 chunks
- Average chunk length: ~177 characters

### Known limitations
- Smaller chunks may help exact retrieval but hurt answer completeness.
- Should be judged by recall/precision against benchmark queries.

---

## 4. `Heading_sections_l2`

### Plain-English definition
Heading-based chunking that tries to split text by document sections, especially level-2 style headings or numbered process sections.

### Current implementation behavior
- Looks for heading-like boundaries:
  - numbered headings
  - all-caps headings
  - colon-ending labels
  - section/process titles
- Groups text under detected headings.

### Parameters
- `heading_level = 2`
- Minimum chunk length: 50 characters
- Output columns: `id`, `pdf_name`, `paragraph`

### Observed output
- 127 chunks
- Average chunk length: ~265 characters

### Known limitations
- PDF extraction can flatten heading hierarchy, so heading detection is approximate.
- Some heading-only chunks may be too short or lack enough answer context.
- Needs manual inspection for noisy table-of-contents chunks.

---

## Candidate additional methods to test

The team suggested trying these and keeping whichever gives better recall:

1. `semantic_split`
2. `fixed_tok1200_ov150`

These should be added as benchmark candidate sheets, then evaluated using retrieval recall.

---

## 5. `semantic_split`

### Plain-English definition
Splits text when the topic appears to shift. It tries to keep semantically related sentences together instead of using only fixed size or headings.

### Proposed implementation
Dependency-light first version:
- Split paragraph into sentences.
- Compare adjacent sentence/topic term overlap.
- Start a new chunk when similarity drops below threshold after minimum chunk size is reached.
- Keep chunks below target size.

Optional stronger version:
- Use sentence embeddings to compare adjacent sentence similarity.
- Split at low-similarity boundaries.

### Parameters
- `min_chars = 120`
- `target_chars = 1200`
- `max_chars = 1500`
- `similarity_threshold = 0.12` for lexical fallback

### Why test it
May improve recall when PDFs mix multiple process topics inside one extracted paragraph.

### Risk
Without true embeddings, lexical semantic splitting is approximate. Evaluate by recall before selecting.

---

## 6. `fixed_tok1200_ov150`

### Plain-English definition
Fixed token window chunking. It creates chunks of about 1200 tokens with 150-token overlap.

### Parameters
- `chunk_tokens = 1200`
- `overlap_tokens = 150`
- Minimum chunk length: 50 characters

### Why test it
Provides a stable baseline independent of headings/entities. Useful when PDF extraction is noisy.

### Risk
Can cut through semantic boundaries. Overlap helps reduce lost context, but embedding cost can increase.

---

## Recommended next step

Generate both additional candidate sheets:

- `semantic_split`
- `fixed_tok1200_ov150`

Then run the same retrieval benchmark for all chunking sheets and compare recall. Select the better of the two candidate methods only after measured retrieval results.
