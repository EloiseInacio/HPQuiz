# Plan to Optimize RAG-Based Question/Answer Generation with a Small Local LLM

## Goal

Build an offline question/answer generation pipeline that produces:

- grounded questions,
- diverse coverage of the corpus,
- acceptable quality despite a small local LLM,
- stable structured outputs,
- controllable runtime.

This plan assumes:

- the LLM is small and weak at long-context reasoning,
- generation happens offline,
- latency is less important than quality,
- the corpus is the Harry Potter books,
- question diversity matters,
- retrieval can be made intentionally stochastic.

---

# 1. Core Design Principles

## 1.1 Use RAG for coverage, not just relevance
Standard RAG often retrieves only the most semantically similar passages, which tends to produce:

- repeated topics,
- repeated characters,
- repeated events,
- repetitive question styles.

For offline quiz generation, retrieval should optimize for:

- diversity,
- corpus coverage,
- fact density,
- answerability.

## 1.2 Use the small LLM only for narrow subtasks
Do not ask the model to:

- read large chunks,
- infer many facts at once,
- generate several questions at once,
- create good distractors from scratch,
- perform formatting and quality control in the same pass.

Instead, decompose generation into small, local steps.

## 1.3 Prefer pipeline complexity over model complexity
Because compute is limited, quality should come from:

- retrieval strategy,
- chunk selection,
- fact extraction,
- templating,
- validation,
- repair.

This is the right tradeoff when generation is offline.

---

# 2. Retrieval Strategy: From Pure Similarity to Controlled Diversity

## 2.1 Why random token querying is useful
Querying the index with randomly selected tokens can help uncover:

- minor characters,
- secondary events,
- less central scenes,
- unusual vocabulary,
- broader thematic coverage.

This is useful because similarity-based retrieval alone will over-focus on the most represented or semantically central concepts.

## 2.2 Main risk of naive random token querying
Pure random token selection can retrieve:

- low-information passages,
- function-word-dominated chunks,
- context-poor fragments,
- repetitive or noisy retrievals.

So the random strategy must be constrained.

## 2.3 Recommended query strategy: guided random retrieval
Instead of querying with arbitrary random tokens, query using **randomly selected informative tokens**.

### Token selection rules
Only sample tokens that are likely to produce meaningful retrieval:

- named entities,
- rare nouns,
- magical objects,
- spells,
- creature names,
- location names,
- faction names,
- distinctive adjectives,
- event-associated verbs.

### Exclude
Do not sample:

- stopwords,
- punctuation,
- extremely common words,
- pronouns,
- generic verbs,
- generic adjectives,
- OCR/noise artifacts.

## 2.4 Better than single-token querying: token sets
Single-token queries may be too weak. Prefer:

- 1 anchor token,
- optionally 1 supporting token,
- optionally a category hint.

### Examples
- `Azkaban`
- `Pensieve`
- `Hippogriff`
- `Sirius + Azkaban`
- `Triwizard + tournament`
- `Patronus + stag`

This preserves randomness while improving precision.

## 2.5 Multi-mode retrieval strategy
Use several query modes rather than one.

### Mode A: random informative token queries
Purpose:
- maximize diversity,
- surface uncommon content.

### Mode B: random entity-pair queries
Purpose:
- retrieve relational facts,
- support medium/hard questions.

### Mode C: random chunk seeding
Purpose:
- sample the corpus directly,
- avoid retrieval bias from embeddings alone.

### Mode D: standard semantic retrieval
Purpose:
- maintain a baseline of high-confidence passages.

### Recommendation
Mix all four modes in a scheduled ratio.

For example:
- 40% random informative token queries,
- 20% entity-pair queries,
- 20% random chunk seeds,
- 20% semantic retrieval.

---

# 3. Build a Retrieval Candidate Pool Before Generation

## 3.1 Do not generate immediately from the first retrieved chunk
The first retrieval result is not necessarily the best generation unit.

Instead, retrieval should produce a **candidate pool**.

## 3.2 Candidate pool structure
For each query, collect:

- top-k retrieved chunks,
- metadata,
- retrieval score,
- source book/chapter if available,
- token/entity statistics,
- novelty score relative to prior selected chunks.

## 3.3 Rank candidates for question potential
After retrieval, score candidate chunks by:

- fact density,
- entity clarity,
- self-containedness,
- novelty,
- answerability,
- low redundancy with already processed chunks.

## 3.4 Recommended chunk selection rule
Select chunks that are:

- self-contained,
- fact-rich,
- not too long,
- not too similar to previously selected chunks,
- centered on one scene or event.

---

# 4. Chunking Strategy for Small-Model Generation

## 4.1 Chunking directly affects question quality
Bad chunking causes:

- incomplete facts,
- pronoun ambiguity,
- weak answerability,
- poor distractors,
- generic questions.

## 4.2 Recommended chunk properties
Chunks should be:

- short,
- semantically coherent,
- event-focused,
- entity-resolved,
- minimally overlapping.

## 4.3 Practical chunking rule
Prefer chunks that contain:

- one main event,
- one explicit interaction,
- one consistent topic,
- clear named entities.

## 4.4 Avoid
Avoid chunks that:

- span too many scenes,
- begin mid-dialogue without context,
- rely heavily on pronouns,
- contain multiple unrelated events.

## 4.5 Optional two-level chunking
Use two parallel chunk views:

### Retrieval chunks
Slightly larger chunks for recall.

### Generation chunks
Smaller sub-chunks extracted from retrieval chunks for the small LLM.

This is often better than using the same chunk for both retrieval and generation.

---

# 5. Compartmentalize Full Question Generation

## 5.1 Why compartmentalization is necessary
A small model performs poorly when asked to do everything at once.

The full generation process should be split into narrow phases.

## 5.2 Recommended offline generation pipeline

### Stage 1: retrieval
Generate a diverse candidate set of chunks.

### Stage 2: chunk filtering
Keep only the chunks that are self-contained and fact-rich.

### Stage 3: fact extraction
Extract atomic facts from one chunk at a time.

### Stage 4: fact normalization
Convert extracted facts into a structured format.

### Stage 5: fact selection
Choose which facts deserve question generation.

### Stage 6: question type assignment
Assign a question format based on fact type.

### Stage 7: answer generation
Generate the correct answer from the fact only.

### Stage 8: distractor generation
Generate distractors separately with typed constraints.

### Stage 9: validation
Check schema, grounding, and uniqueness.

### Stage 10: repair
Fix only the failed part.

### Stage 11: deduplication
Remove duplicate or near-duplicate items.

### Stage 12: ranking
Keep only the best final questions.

---

# 6. Fact Extraction as the Main Bottleneck Fix

## 6.1 Do not generate questions directly from retrieved text
The small model should first extract facts.

## 6.2 Target fact shape
Each fact should be:

- atomic,
- explicit,
- answerable,
- unambiguous,
- grounded in one chunk.

## 6.3 Example structured fact
Use a simple structure like:

- `fact_id`
- `subject`
- `relation`
- `object`
- `book`
- `source_chunk_id`
- `fact_type`
- `confidence`

## 6.4 Benefits
Fact extraction reduces model load by converting:

- messy text
into
- structured, question-ready units.

## 6.5 Practical rule
Generate at most a small number of facts per chunk, such as:

- 3 to 5 facts per chunk.

That is easier to validate than attempting broad extraction.

---

# 7. Question Type Routing

## 7.1 Do not let the model invent the question format every time
Question type should be assigned by rules or lightweight classification.

## 7.2 Example question types
Map fact types to formats such as:

- who
- what
- where
- when
- which object
- which spell
- which character
- which creature
- true/false
- multiple choice
- relation matching

## 7.3 Example routing logic
If fact type is:

- `character-action` -> who/what-did-X-do
- `object-use` -> which object
- `location-event` -> where did X happen
- `spell-effect` -> which spell caused X
- `book-event` -> in which book did X occur

## 7.4 Benefit
Routing reduces open-ended burden on the LLM and improves consistency.

---

# 8. Distractor Generation Must Be Separate

## 8.1 Why distractors are hard
Distractors are one of the hardest parts for a small model.

Typical failures:

- obviously wrong answers,
- duplicate answers,
- wrong type,
- nonsense distractors.

## 8.2 Solution
Generate distractors in a dedicated step.

## 8.3 Typed distractor banks
Maintain static or extracted banks for:

- characters,
- spells,
- houses,
- creatures,
- locations,
- magical objects,
- professors,
- villains,
- books.

## 8.4 Generation rule
Distractors should be:

- same semantic type as the answer,
- plausible,
- wrong,
- distinct from each other,
- not too close to the correct answer wording.

## 8.5 Best practice
Prefer hybrid distractor generation:

- rule-based candidate pool,
- LLM selects or lightly edits.

This is much safer than fully free-form distractor generation.

---

# 9. Offline Scheduling Strategy

## 9.1 Since generation is offline, use batch planning
Because runtime is acceptable, use a multi-pass process.

## 9.2 Suggested schedule

### Pass 1: corpus analysis
Build:

- token frequencies,
- entity lists,
- chunk metadata,
- chunk novelty map.

### Pass 2: retrieval candidate generation
Generate many candidate chunks through all retrieval modes.

### Pass 3: fact extraction
Extract and store structured facts.

### Pass 4: question generation
Generate one question per selected fact.

### Pass 5: validation and repair
Filter and repair failures.

### Pass 6: ranking and export
Produce the final dataset.

## 9.3 Benefit
This makes the system easier to debug, cache, and improve incrementally.

---

# 10. Cache Everything

## 10.1 Offline systems should avoid recomputing expensive steps
Cache:

- tokenized chunks,
- query results,
- retrieval candidate pools,
- extracted facts,
- generated questions,
- validation scores,
- repair outputs.

## 10.2 Why this matters
With a small local model, total compute is precious even if latency is not.

Caching enables:

- experimentation,
- comparison of prompt versions,
- prompt repair without re-running retrieval,
- ablation studies.

---

# 11. Recommended Retrieval Query Generator

## 11.1 Build a token inventory
From the corpus, build a filtered token inventory with metadata:

- token text,
- frequency,
- IDF-like rarity score,
- POS tag,
- entity type if any,
- books/chapters where it appears,
- co-occurring entities.

## 11.2 Sampling strategy
Sample queries from this inventory with weighted randomness.

### Good weighting factors
Increase probability for tokens that are:

- informative,
- not too common,
- not too rare,
- entity-linked,
- spread across the corpus.

### Lower probability for tokens that are:
- too frequent,
- too noisy,
- too local,
- low semantic value.

## 11.3 Recommended query templates
Generate retrieval queries using templates like:

- single informative token
- entity + entity
- entity + action
- object + location
- spell + effect
- creature + event
- faction + person

## 11.4 Novelty-aware sampling
Track which concepts have already been used.

Decrease future sampling probability for:

- overused entities,
- overused books,
- repeated scenes,
- repeated answer types.

This is essential for diversity.

---

# 12. Diversity Control Framework

## 12.1 Diversity should be explicit, not accidental
Track diversity across:

- books,
- chapters,
- characters,
- locations,
- spells,
- objects,
- creatures,
- question types,
- difficulty levels.

## 12.2 Maintain quotas
Example quotas:

- no single character dominates more than a threshold,
- each book contributes a minimum number of questions,
- question types stay balanced,
- answer types stay balanced.

## 12.3 Use a novelty score
Each candidate question should receive a novelty score based on:

- entity overlap,
- lexical overlap,
- template overlap,
- same source chunk reuse,
- same answer reuse.

Select high-quality but also high-novelty items.

---

# 13. Validation Strategy

## 13.1 Validation is mandatory with a small model
Never trust raw outputs.

## 13.2 Validate at three levels

### Level 1: schema validation
Check:

- required fields,
- JSON validity,
- distractor count,
- uniqueness.

### Level 2: factual validation
Check:

- answer is supported by the source fact,
- question is answerable from the fact,
- distractors are incorrect.

### Level 3: quality validation
Check:

- grammar,
- specificity,
- clarity,
- no trivial leakage,
- no duplication.

## 13.3 Recommended validator mix
Use:

- rule-based checks first,
- lightweight LLM self-check second,
- optional ranking score third.

This keeps cost manageable.

---

# 14. Repair Strategy

## 14.1 Do not regenerate the whole item if only one part fails
Repair should be local.

## 14.2 Example repair actions
If the issue is:

- bad distractors -> regenerate distractors only
- unclear wording -> rewrite question only
- wrong answer type -> reroute question type
- schema failure -> reformat only
- unsupported fact -> discard item

## 14.3 Benefit
Local repair is cheaper and more stable than regeneration from scratch.

---

# 15. Recommended Data Structures

## 15.1 Chunk record
Each chunk should store:

- chunk_id
- source_book
- source_location
- chunk_text
- tokens
- entities
- embeddings
- length
- novelty stats

## 15.2 Fact record
Each extracted fact should store:

- fact_id
- chunk_id
- subject
- relation
- object
- fact_type
- confidence
- entity_types
- support_span if available

## 15.3 Question record
Each generated item should store:

- question_id
- fact_id
- question_type
- question_text
- correct_answer
- distractors
- difficulty
- validation_status
- repair_history
- novelty_score

These structures make debugging much easier.

---

# 16. Suggested Prompt Design Rules

## 16.1 Keep prompts short
Small models perform better with:

- short instructions,
- one task at a time,
- strict output requirements.

## 16.2 General prompt rules
Each prompt should:

- define one task only,
- restrict to one fact or one chunk,
- demand structured output,
- forbid outside knowledge,
- forbid quoting the books,
- avoid chain complexity.

## 16.3 Good prompt sequence
Use separate prompts for:

- fact extraction,
- fact typing,
- question wording,
- answer generation,
- distractor generation,
- validation,
- repair.

---

# 17. Recommended End-to-End Offline Pipeline

## Phase A: corpus preprocessing
- clean text
- chunk text
- extract tokens/entities
- build embeddings
- build token inventory

## Phase B: candidate retrieval
- run guided random token queries
- run entity-pair queries
- run random chunk seed selection
- run standard semantic retrieval
- merge and rank candidates

## Phase C: fact extraction
- extract atomic facts
- normalize and type facts
- remove weak facts

## Phase D: question generation
- assign question type
- generate one question per fact
- generate answer
- generate distractors

## Phase E: validation and repair
- schema validation
- factual validation
- quality validation
- targeted repair
- deduplication

## Phase F: dataset balancing
- enforce diversity quotas
- enforce difficulty balance
- rank and export final questions

---

# 18. Practical First Implementation

## 18.1 First version
Start simple:

- filtered random token queries,
- top-k candidate chunk pool,
- fact extraction,
- one question per fact,
- separate distractor generation,
- rule-based validation,
- deduplication.

## 18.2 Second version
Add:

- novelty-aware retrieval,
- entity-pair queries,
- typed distractor banks,
- repair passes,
- diversity quotas.

## 18.3 Third version
Add:

- candidate scoring,
- better fact typing,
- difficulty balancing,
- template families,
- question ranking.

---

# 19. Main Recommendations in One Place

## Retrieval
- use guided random informative token queries
- mix random and semantic retrieval
- track novelty and coverage
- build candidate pools before generation

## Generation
- never do everything in one prompt
- extract facts first
- generate one question per fact
- route by question type
- separate distractor generation

## Quality control
- validate aggressively
- repair locally
- deduplicate
- enforce diversity quotas

## Runtime strategy
- cache everything
- use offline multi-pass processing
- prioritize stability over speed

---

# 20. Final Takeaway

A small local LLM can still produce good offline question/answer datasets if the system is designed correctly.

The most important shifts are:

1. move from pure similarity retrieval to diversity-aware retrieval
2. use guided random informative token queries instead of naive random tokens
3. split question generation into narrow stages
4. generate from structured facts, not directly from raw chunks
5. validate and repair every item before keeping it

## One-sentence summary

> With a small local LLM, question quality will come much more from retrieval design, fact extraction, and validation than from the generation model alone.