"""Prompt templates for RAG pipeline components."""

__all__ = [
    "RAG_ANSWER_TEMPLATE",
    "INTENT_ROUTER_PROMPT",
    "RETRIEVAL_GRADER_PROMPT",
    "QUERY_REWRITER_PROMPT",
]


# RAG Answer Template
# Purpose: Generate natural language answers from retrieved context
# Usage: Used in the API endpoint to provide final answers to user queries
# Parameters: {context} - retrieved documents, {question} - user query
RAG_ANSWER_TEMPLATE = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Provide a comprehensive, factual answer structured appropriately for the query type:
- For FACT queries (e.g., lists, rankings): Use numbered or bulleted lists with key details.
- For CONCEPT queries (e.g., explanations): Use sections with headings, definitions, and examples.
- For COMPARISON queries (e.g., differences): Use side-by-side comparisons, tables if suitable, or structured pros/cons.
- Include relevant code examples where they illustrate key points (keep under 10 lines).
- Aim for 200-400 words, focusing on clarity and informativeness.

Guidelines for synthesis:
- If multiple context snippets provide information about the same entity (e.g., the same programming language), merge that information into a single, cohesive entry. Do not repeat entities in lists.
- Only include information present in the context. If the context is repetitive, ignore the redundancy and provide a clean summary.
- Prioritize unique facts over multiple mentions of the same fact.

Answer:"""

# Intent Router Prompt
# Purpose: Classify user queries into intent categories for adaptive retrieval
# Usage: Used in intent_router() to determine BM25/Chroma weight ratios
# Categories: FACT (keyword-heavy), CONCEPT (balanced), COMPARISON (semantic-heavy)
# Parameters: {query} - user query to classify
INTENT_ROUTER_PROMPT = """Classify this query into ONE category:

FACT: Asks for a specific list, ranking, or enumerated facts
  Examples: "What are the top 10 languages?", "List languages with static typing"

CONCEPT: Asks for explanation or understanding of a concept
  Examples: "What is functional programming?", "Explain garbage collection"

COMPARISON: Asks to compare or contrast multiple items
  Examples: "Compare Go and Rust", "What are differences between Python and Java?"

Query: {query}

Answer with only the category name (FACT, CONCEPT, or COMPARISON):"""

# Retrieval Grader Prompt
# Purpose: Evaluate if retrieved documents are relevant to the query
# Usage: Used in retrieval_grader() to decide whether to proceed or rewrite query
# Logic: Lenient grading - YES if any document mentions topic, NO only if all unrelated
# Parameters: {query} - user query, {formatted_docs} - retrieved documents as formatted text
RETRIEVAL_GRADER_PROMPT = """You are a grader for retrieval quality. Judge if these documents are relevant to the query.

Query: {query}

Retrieved Documents:
{formatted_docs}

GRADING RULES:
1. If ANY document mentions the query topic → ANSWER: YES
2. If ANY document contains information related to the query → ANSWER: YES
3. Only answer NO if ALL documents are completely unrelated to the query topic

INSTRUCTION: This is a lenient grading. Most queries should receive YES unless the documents are obviously wrong.

Answer with ONLY the word YES or NO, nothing else:"""

# Query Rewriter Prompt
# Purpose: Enhance queries that failed retrieval to improve document matching
# Usage: Used in query_rewriter() when retrieval_grader() returns NO
# Strategy: Add specific keywords and context to make queries more searchable
# Parameters: {query} - original query that failed retrieval
QUERY_REWRITER_PROMPT = """Your previous retrieval for this query didn't return relevant documents:
Original Query: {query}

Rewrite this query to be more explicit, adding keywords that might be in a list or ranking.

Examples:
- "What are the top 10 languages?" → "TIOBE index top 10 most popular programming languages ranking list"
- "Compare Go and Rust" → "Go versus Rust comparison features differences systems programming"

Rewritten Query:"""
