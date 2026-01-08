"""Prompt templates for RAG pipeline components."""

__all__ = [
    "INTENT_ROUTER_PROMPT",
    "JUDGE_PROMPT",
    "QUERY_REWRITER_PROMPT",
    "RAG_ANSWER_TEMPLATE",
    "RETRIEVAL_GRADER_PROMPT",
]


# RAG Answer Template
# Purpose: Generate natural language answers from retrieved context
# Usage: Used in the API endpoint to provide final answers to user queries
# Parameters: {context} - retrieved documents, {question} - user query
RAG_ANSWER_TEMPLATE = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. ONLY use information explicitly stated in the context above. Do NOT add any external knowledge.
2. If the context does not contain enough information to fully answer the question, say "Based on the provided context, I cannot find specific information about [topic]."
3. Every claim in your answer MUST be directly supported by text in the context.
4. Do NOT infer, assume, or extrapolate beyond what is written in the context.
5. If you're unsure whether something is in the context, do NOT include it.
6. NEVER use your training knowledge to fill gaps. If the context doesn't say it, don't say it.

Formatting guidelines:
- For lists/rankings: Use numbered or bulleted lists.
- For explanations: Use clear sections with the key points from the context.
- For comparisons: Structure as side-by-side points from the context.
- Keep answers focused (150-300 words) and quote context directly where possible.

Synthesis rules:
- Merge information about the same entity from multiple context snippets.
- Only include information present in the context. Ignore redundancy.
- Prefer direct quotes or close paraphrases over summaries.

Answer (grounded strictly in the context above):"""

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
# Also handles synthesis queries by expanding into multiple search terms
# Parameters: {query} - original query that failed retrieval
QUERY_REWRITER_PROMPT = """Your previous retrieval for this query didn't return relevant documents:
Original Query: {query}

Rewrite this query to be more explicit. For comparisons or synthesis questions, include BOTH concepts as separate searchable terms.

Examples:
- "What are the top 10 languages?" → "TIOBE index top 10 most popular programming languages ranking list"
- "Compare Go and Rust" → "Go programming language features performance Rust programming language features comparison"
- "differences between interpreted and compiled" → "interpreted languages definition characteristics compiled languages definition advantages disadvantages"
- "What is C# used for?" → "C# programming language applications use cases .NET framework"

Tips:
- Include synonyms and related terms
- For comparison queries, mention BOTH items being compared
- Add domain-specific keywords (e.g., "programming language", "framework")

Rewritten Query:"""

# Judge Prompt
# Purpose: Evaluate RAG answers for correctness, relevance, and faithfulness
# Usage: Used in evaluate CLI command to score generated answers
# Parameters: {question} - user query, {context} - retrieved documents, {answer} - generated answer
JUDGE_PROMPT = """Evaluate this AI answer. Output ONLY JSON with scores 1-5.

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}

SCORING GUIDE:
- correctness: 5=fully accurate, 4=minor issues, 3=some errors, 2=major errors, 1=wrong
- relevance: 5=directly answers question, 4=mostly relevant, 3=partially relevant, 2=barely relevant, 1=off-topic
- faithfulness: 5=every claim appears in context, 4=1 minor addition, 3=some facts not in context, 2=many facts not in context, 1=mostly external knowledge

FAITHFULNESS CHECK: Compare each claim in the ANSWER to the CONTEXT. If a claim cannot be traced to specific text in the context, reduce the faithfulness score.

JSON only (replace X with your 1-5 scores):
{{"correctness":{{"score":X,"explanation":"brief reason"}},"relevance":{{"score":X,"explanation":"brief reason"}},"faithfulness":{{"score":X,"explanation":"brief reason"}}}}"""
