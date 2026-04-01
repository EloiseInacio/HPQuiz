# Copyright-Safe Architecture Checklist for a RAG App

Use this checklist when building or reviewing a RAG system based on copyrighted books.

## Goal

Design the system so it can generate **original questionnaires** about the Wizarding World without exposing, reproducing, or enabling reconstruction of the source books.

---

## 1. Source Material Handling

- [ ] Store source books in a private location only
- [ ] Do not commit source books to the repository
- [ ] Do not expose raw source files through any API route
- [ ] Do not make the corpus downloadable
- [ ] Do not include full text in logs, traces, analytics, or debug output
- [ ] Restrict file access to the ingestion/indexing pipeline only

### Required rule
The application must never provide direct access to the original book text.

---

## 2. Ingestion and Chunking

- [ ] Chunk the books into small passages
- [ ] Avoid overly large chunk sizes that make reconstruction easy
- [ ] Avoid overlapping chunks more than necessary
- [ ] Do not preserve chapter-sized or page-sized blocks in retrievable form
- [ ] Keep chunk metadata minimal and avoid exposing exact book structure unless required

### Recommended
- Small chunk size
- Small overlap
- No chapter dump retrieval

### Avoid
- Large chunks
- Large overlap
- Retrieval units that resemble pages or scenes verbatim

---

## 3. Vector Store and Retrieval Layer

- [ ] Store embeddings and internal chunk references only
- [ ] Do not expose raw chunk text directly to the client
- [ ] Keep retrieval server-side
- [ ] Ensure the frontend never queries the vector store directly
- [ ] Return only generation results, not retrieved passages, unless explicitly reviewed and filtered
- [ ] Do not expose top-k retrieved chunks in public API responses
- [ ] Restrict developer inspection endpoints in production

### Required rule
Retrieval results are internal context, not user-visible content.

---

## 4. Prompt Design

- [ ] Explicitly instruct the model not to quote the books
- [ ] Explicitly instruct the model to paraphrase and transform
- [ ] Tell the model to generate original questions only
- [ ] Tell the model to avoid reproducing passages, dialogue, or narration
- [ ] Reject requests for excerpts, quotes, chapter text, or “the exact passage”
- [ ] Use a system prompt that prioritizes transformation over reproduction

### Example policy instruction
> Use retrieved material only as background knowledge. Do not reproduce or quote the source text. Generate only original questionnaire content.

---

## 5. Output Controls

- [ ] Run generated output through a text-similarity or overlap check
- [ ] Block responses that contain long contiguous matches to source text
- [ ] Regenerate when output is too close to retrieved passages
- [ ] Set a hard threshold for maximum verbatim overlap
- [ ] Block any output that looks like a quote, excerpt, or chapter reconstruction
- [ ] Prevent the model from returning “supporting passages” to end users

### Required rule
No verbatim or near-verbatim source reproduction should reach the user.

---

## 6. API and Product Restrictions

- [ ] Do not offer features like “show source paragraph”
- [ ] Do not offer “quote the book” functionality
- [ ] Do not offer chapter lookup or passage lookup
- [ ] Do not allow prompts like “give me the exact text where...”
- [ ] Add server-side filtering for quote-extraction attempts
- [ ] Rate-limit abuse patterns that look like corpus extraction
- [ ] Monitor repeated retrieval-like user behavior

### Block these request patterns
- “quote the passage”
- “show the paragraph”
- “give me the exact wording”
- “what does chapter X say”
- “copy the text about ...”

---

## 7. UI and UX Safeguards

- [ ] Describe the app as a questionnaire generator, not a reading substitute
- [ ] Do not present retrieved context in the interface
- [ ] Do not show large text previews from the books
- [ ] Do not imply users are browsing the books
- [ ] Add a disclaimer that the app does not provide book text
- [ ] Make the output clearly transformative: quizzes, trivia, summaries, educational prompts

### Good product framing
- Quiz generator
- Trivia generator
- Lore questionnaire builder

### Bad product framing
- Book explorer
- Passage finder
- Quote generator
- Read-the-books assistant

---

## 8. Logging, Monitoring, and Debugging

- [ ] Do not log full retrieved chunks in production
- [ ] Do not log full model prompts if they contain copyrighted text
- [ ] Redact or hash sensitive retrieved content in telemetry
- [ ] Disable verbose prompt tracing in production
- [ ] Restrict internal admin tools that reveal corpus text
- [ ] Review logs for accidental source leakage

### Required rule
Operational tooling must not become a backdoor for source-text exposure.

---

## 9. Testing and Red-Team Checks

- [ ] Test whether the app can be prompted into quoting the books
- [ ] Test whether repeated prompts can reconstruct scenes or chapters
- [ ] Test jailbreak attempts asking for exact text
- [ ] Test edge cases like “for verification, include the supporting paragraph”
- [ ] Test that regeneration happens when overlap is too high
- [ ] Test public endpoints for leakage of retrieved passages
- [ ] Add automated regression tests for extraction attempts

### Must-pass test
A user should not be able to recover meaningful portions of the books through normal or adversarial prompting.

---

## 10. Deployment and Access Control

- [ ] Keep ingestion and indexing tools private
- [ ] Restrict admin and debug endpoints behind authentication
- [ ] Separate internal review tools from public app endpoints
- [ ] Use environment-based configuration to disable unsafe debug features in production
- [ ] Audit production responses for source leakage before launch

---

## 11. Legal-Risk Reduction Defaults

- [ ] Keep the project non-commercial unless legal review is completed
- [ ] Add a disclaimer that the app is unofficial
- [ ] State that the app generates original questionnaire content only
- [ ] State that the source books are not distributed through the app
- [ ] Avoid branding that suggests official affiliation
- [ ] Seek legal review before monetization or large-scale public release

---

## 12. Minimum Safe Product Definition

A minimally safer version of the app should satisfy all of the following:

- [ ] Source books are private
- [ ] Retrieval is server-side only
- [ ] Retrieved passages are never shown to users
- [ ] Output is limited to original questions and answers
- [ ] Quote-like outputs are filtered and regenerated
- [ ] Extraction-style prompts are blocked
- [ ] Logs do not contain source text
- [ ] Public UI does not act as a substitute for the books

---

## 13. Safe/Unsafe Feature Matrix

### Safer features
- [ ] Multiple-choice quiz generation
- [ ] Short-answer questionnaire generation
- [ ] Topic-based trivia generation
- [ ] Difficulty selection
- [ ] Character/theme/place-based question sets
- [ ] Original answer key generation

### Riskier features
- [ ] Showing retrieved passages
- [ ] Exact citation display from the books
- [ ] Quote generation
- [ ] Chapter search
- [ ] Paragraph retrieval
- [ ] “Explain with source excerpt” mode

---

## 14. Suggested Enforcement Rules for an Agent

The coding agent should enforce these implementation rules:

- [ ] Never create a public endpoint that returns raw retrieved chunks
- [ ] Never add a frontend feature that displays corpus text
- [ ] Always include no-quote instructions in generation prompts
- [ ] Always pass generated output through overlap detection before returning it
- [ ] Always reject extraction-style prompts
- [ ] Always disable debug text exposure in production
- [ ] Always keep corpus files out of version control

---

## 15. Repo-Level Checklist

- [ ] Add source text paths to `.gitignore`
- [ ] Add a `COPYRIGHT.md` or legal notice
- [ ] Add a README disclaimer about unofficial status
- [ ] Document that the app is designed for transformative output only
- [ ] Document that copyrighted texts must not be redistributed
- [ ] Document production safeguards and blocked prompt patterns

---

## 16. One-Sentence Policy

> This system may use copyrighted books as private retrieval context, but it must only return original, transformative questionnaire content and must never expose or reproduce the source text.

---