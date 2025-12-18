"""Router and grader prompts for intent-based adaptive RAG."""

INTENT_ROUTER_PROMPT = """Classify this query into ONE category:

FACT: Asks for a specific list, ranking, or enumerated facts
  Examples: "What are the top 10 languages?", "List languages with static typing"

CONCEPT: Asks for explanation or understanding of a concept
  Examples: "What is functional programming?", "Explain garbage collection"

COMPARISON: Asks to compare or contrast multiple items
  Examples: "Compare Go and Rust", "What are differences between Python and Java?"

Query: {query}

Answer with only the category name (FACT, CONCEPT, or COMPARISON):"""

RETRIEVAL_GRADER_PROMPT = """You are a grader for retrieval quality.

Query: {query}

Retrieved Documents:
{formatted_docs}

Are these documents relevant to the query? They don't need to fully answer it, just be on-topic.

If the query asks about a programming language or technology, documents about that topic are relevant.
If the query asks for a list, documents mentioning any items from that domain are relevant.
If the query asks for comparison, documents about either subject are relevant.

Be lenient - if documents are even partially on-topic, say YES.

Answer with only YES or NO:"""

QUERY_REWRITER_PROMPT = """Your previous retrieval for this query didn't return relevant documents:
Original Query: {query}

Rewrite this query to be more explicit, adding keywords that might be in a list or ranking.

Examples:
- "What are the top 10 languages?" → "TIOBE index top 10 most popular programming languages ranking list"
- "Compare Go and Rust" → "Go versus Rust comparison features differences systems programming"

Rewritten Query:"""
