# Improving Question Generation Quality with Qwen2.5-1.5B-Instruct in a RAG Quiz App

## Problem Summary

Qwen2.5-1.5B-Instruct is lightweight and efficient, but for questionnaire generation it often produces:

- vague questions  
- repetitive phrasing  
- trivial or overly obvious questions  
- hallucinated details  
- malformed multiple-choice answers  
- poor difficulty control  
- weak grounding in retrieved context  

This is expected for a small model. The solution is not to rely on the model to do everything in one step.

---

## Core Strategy

Use the model for small, constrained subtasks, not for full end-to-end generation.

> Reduce generation difficulty by moving quality control into pipeline design.

---

## 1. Multi-Step Generation

### Problem
Single prompts overload the model.

### Solution
Split into stages:

1. Retrieve context  
2. Extract facts  
3. Select facts  
4. Generate question  
5. Generate answer  
6. Generate distractors  
7. Validate  
8. Repair  

---

## 2. Generate from Facts

### Example Fact Format
{  
  "subject": "Sorting Hat",  
  "predicate": "assigned Harry Potter to",  
  "object": "Gryffindor"  
}

### Benefit
- reduces hallucination  
- improves grounding  

---

## 3. Reduce Context Size

- Use 1–3 chunks max  
- Deduplicate overlaps  
- Remove irrelevant metadata  

---

## 4. Improve Chunking

- Split by events or scenes  
- Ensure entity clarity  
- Avoid pronoun-only chunks  

---

## 5. Enforce Output Schema

Example schema:

{  
  "question": "...",  
  "answer": "...",  
  "distractors": ["...", "...", "..."],  
  "explanation": "...",  
  "source_fact": "..."  
}

---

## 6. One Question per Call

- 1 fact → 1 question  
- Avoid batch generation  

---

## 7. Strong Prompt Constraints

Example:

Use only the provided fact.  
Do not quote the books.  
Write one clear MCQ.  
Return JSON only.  

---

## 8. Template-Based Questions

Examples:

- Who did X?  
- Where did X happen?  
- What object was used?  

---

## 9. Separate Distractor Generation

Use type-aware distractors:

Character → other characters  
Spell → other spells  
House → other houses  

---

## 10. Validation Rules

Reject if:

- duplicate distractors  
- answer obvious in question  
- unsupported by fact  
- grammatical issues  

---

## 11. Candidate Ranking

- Generate 2–4 candidates  
- Score and keep best  

---

## 12. Self-Check

Verification prompt:

Return PASS or FAIL:  
- Is answer correct?  
- Are distractors wrong?  

---

## 13. Prefer Classification Tasks

Instead of generating:
→ select from candidates  

---

## 14. Entity Banks

Maintain lists:

- characters  
- spells  
- locations  
- creatures  

---

## 15. External Difficulty Control

Define difficulty outside the model:

- Easy → direct facts  
- Medium → relationships  
- Hard → indirect or multi-step reasoning  

---

## 16. Lightweight Validation Pipeline

Check:

- schema validity  
- answer correctness  
- distractor uniqueness  
- grounding in fact  

Reject or repair failed items.

---

## 17. Repair Instead of Regenerate

Example:

Repair this MCQ.  
Keep the correct answer.  
Fix distractors only.  

---

## 18. Deduplication

Remove duplicates using:

- normalized text  
- answer overlap  
- semantic similarity (optional)  

---

## 19. Hybrid System Design

Use rules for:

- retrieval  
- typing  
- templates  
- validation  

Use LLM for:

- phrasing  
- light rewriting  

---

## 20. Minimal High-Quality Pipeline

1. retrieve 1–2 chunks  
2. extract facts  
3. classify fact type  
4. choose template  
5. generate question  
6. generate answer  
7. generate distractors  
8. validate  
9. repair  
10. deduplicate  

---

## 21. Priority Improvements

High impact:

- multi-step pipeline  
- fact-based generation  
- small context  
- validation  

Medium:

- ranking  
- templates  
- entity banks  

Advanced:

- fine-tuning  
- synthetic dataset  

---

## 22. Key Takeaway

Small model quality comes from:

- structure  
- constraints  
- validation  
- decomposition  

Not from longer prompts.

---

## 23. One-Line Rule

> Use the model as a component, not as the system.