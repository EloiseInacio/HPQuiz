# Wizarding World Questionnaire Generator

A Retrieval-Augmented Generation (RAG) application built on the *Harry Potter* book series to generate questionnaires about the Wizarding World.

## Overview

This project uses a Retrieval-Augmented Generation pipeline to create lore-grounded questionnaires based on the *Harry Potter* books. Instead of relying only on a language model’s parametric knowledge, the system retrieves relevant passages from the source corpus and uses them to generate questions that are more faithful to the books.

The current repository contains the RAG backend only. The long-term goal is to integrate this backend into a web application where users can generate themed questionnaires on characters, spells, magical creatures, places, events, and broader Wizarding World knowledge.

## Motivation

Generating high-quality questionnaires on a fictional universe is challenging because factual consistency matters. In the case of the Wizarding World, questions should reflect the content of the books rather than vague or hallucinated model knowledge.

A RAG-based approach helps by:

- grounding question generation in the original text,
- improving factual accuracy,
- enabling topic-specific questionnaire creation,
- making the generation process more transparent and extensible.

## Project Goal

The intended web application will allow users to generate custom questionnaires about the Wizarding World, such as:

- multiple-choice quizzes on Hogwarts houses,
- short-answer questions about major plot events,
- character-based trivia sets,
- thematic questionnaires on magic, creatures, or locations,
- difficulty-adaptive educational or entertainment quizzes.

At present, only the retrieval-augmented generation pipeline is implemented in this repository.

## Current Status

### Implemented
- RAG pipeline over the *Harry Potter* books
- document indexing and retrieval
- context-grounded question generation

### Not yet implemented
- web interface
- user authentication
- quiz customization UI
- answer validation and scoring
- persistent questionnaire storage
- deployment-ready frontend/backend integration

## How It Works

The pipeline follows a standard RAG workflow:

1. **Corpus ingestion**  
   The *Harry Potter* books are processed and split into chunks.

2. **Embedding and indexing**  
   The chunks are transformed into vector embeddings and stored in a vector database or retrieval index.

3. **User query or generation prompt**  
   A prompt defines the target questionnaire scope, such as a topic, difficulty, or style.

4. **Retrieval**  
   Relevant passages are retrieved from the indexed corpus.

5. **Question generation**  
   A language model uses the retrieved passages as context to generate grounded questionnaire items.

This architecture makes it possible to produce questions that are tied more closely to the source material.

## Example Use Cases

- Generate a beginner-level quiz on Hogwarts professors
- Create a questionnaire about magical objects and artifacts
- Build a trivia set focused on *Harry Potter and the Prisoner of Azkaban*
- Produce lore-grounded study material for fans or readers
- Support educational exploration of narrative content through question generation

## Suggested Future Architecture

A complete web app version could include:

- **Frontend:** React, Next.js, or another modern web framework
- **Backend:** Python API with FastAPI or Flask
- **Vector Store:** FAISS, Chroma, Weaviate, Pinecone, or similar
- **LLM Layer:** OpenAI API or another compatible language model provider
- **Storage:** database for generated quizzes and user sessions

## Limitations

- The current project does not yet expose a user-facing application
- Output quality depends on corpus preprocessing and retrieval 

## Disclaimer

This project is a technical and experimental implementation for questionnaire generation based on the *Harry Potter* books. It is not affiliated with, endorsed by, or associated with J.K. Rowling, Warner Bros., or any official Wizarding World entity.
