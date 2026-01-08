"""CLI for ragchain."""

import asyncio

import click

from ragchain.config import config
from ragchain.ingestion.loaders import load_tiobe_languages, load_wikipedia_pages
from ragchain.ingestion.storage import ingest_documents


@click.group()
def cli():
    """RAG pipeline CLI."""
    pass


@cli.command()
@click.option("--n", default=10, help="Number of languages to ingest")
def ingest(n):
    """Ingest programming language documents into local vector store.

    Fetches top-n from TIOBE, loads Wikipedia articles, splits them,
    and stores in Chroma for semantic search.

    Args:
        n: Number of languages to ingest (default: 10)
    """

    async def _ingest():
        click.echo(f"Fetching top {n} languages from TIOBE...")
        langs = await load_tiobe_languages(n)
        click.echo(f"Fetched {len(langs)} languages: {', '.join(langs)}")

        click.echo("Loading Wikipedia pages...")
        docs = await load_wikipedia_pages(langs)
        click.echo(f"Loaded {len(docs)} documents from languages: {', '.join({d.metadata.get('language', 'Unknown') for d in docs})}")

        click.echo("Ingesting into vector store...")
        result = await ingest_documents(docs)
        click.echo(f"Result: {result}")

    asyncio.run(_ingest())


@cli.command()
@click.argument("query")
@click.option("--k", default=4, help="Number of results")
def search(query, k):
    """Search ingested documents using semantic similarity.

    Args:
        query: Search query (positional argument)
        k: Number of results to return (default: 4)
    """

    async def _search():
        from ragchain.inference.rag import search as search_func

        result = await search_func(query, k=k)
        click.echo(f"Query: {result['query']}")
        for i, res in enumerate(result["results"], 1):
            metadata = res.get("metadata", {})
            title = metadata.get("title", "Unknown") if isinstance(metadata, dict) else "Unknown"
            click.echo(f"\n{i}. {title}")
            click.echo(f"   {res['content'][:200]}...")

    asyncio.run(_search())


@cli.command()
@click.argument("query")
@click.option("--model", default=config.ollama_model)
def ask(query, model):
    """Ask a question and get an answer using RAG + LLM.

    Uses the RAG pipeline to retrieve relevant documents and generate an answer.

    Args:
        query: Question to ask (positional argument)
        model: LLM model to use for generation (default: config.ollama_model)
    """

    async def _ask():
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import OllamaLLM

        from ragchain.graph import rag_graph
        from ragchain.prompts import RAG_ANSWER_TEMPLATE

        click.echo("Retrieving relevant documents...")

        initial_state = {
            "query": query,
            "intent": "CONCEPT",
            "retrieved_docs": [],
            "retrieval_grade": "NO",
            "rewritten_query": "",
            "retry_count": 0,
        }

        final_state = rag_graph.invoke(initial_state)
        retrieved_docs = final_state["retrieved_docs"]

        if not retrieved_docs:
            click.echo("No relevant documents found.")
            return

        click.echo(f"Found {len(retrieved_docs)} documents. Generating answer...")

        llm = OllamaLLM(model=model, base_url=config.ollama_base_url, temperature=0.1, num_ctx=config.ollama_gen_ctx)
        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=query))

        click.echo(f"\nQ: {query}")
        click.echo(f"A: {answer}")

    asyncio.run(_ask())


@cli.command()
@click.option("--model", default=config.ollama_model, help="LLM model to use for generation and judging")
def evaluate(model):
    """Evaluate RAG answers for example questions using LLM-as-judge.

    Runs through example questions, generates answers using the RAG pipeline,
    and evaluates correctness, relevance, and faithfulness using an LLM judge.

    Args:
        model: LLM model to use for generation and judging (default: config.ollama_model)
    """
    questions = [
        # --- Original 7 for demo purposes ---
        "What is Python used for?",
        "Compare Go and Rust for systems programming",
        "What are the key features of functional programming in Haskell?",
        "How has Java evolved since its release?",
        "What are the main differences between interpreted and compiled languages?",
        "Which languages are commonly used for machine learning?",
        "What are the top 10 most popular languages?",
        # --- New additions for thorough coverage ---
        "How does TypeScript differ from JavaScript?",
        "What is the primary purpose of C# and the .NET framework?",
        "Why is C still preferred over C++ for embedded systems?",
        "What are the main use cases for PHP in modern web development?",
        "Why is SQL classified as a domain-specific language?",
        "Compare Swift and Objective-C for iOS development",
        "What role does Ruby on Rails play in web development?",
        "How does R differ from Python for statistical analysis?",
        "Why did Google adopt Kotlin as the preferred language for Android?",
        "Why is Fortran still used in scientific computing?",
        "What are the primary industries that still use COBOL?",
        "What makes Scratch distinct from text-based programming languages?",
        "Why is Ada used in safety-critical systems like aerospace?",
    ]

    click.echo(f"Evaluating {len(questions)} questions...")

    async def _evaluate():
        from ragchain.generation.judge import evaluate_questions

        # Run evaluations
        for i, question in enumerate(questions, 1):
            click.echo(f"\n[{i}/{len(questions)}] {question}")

        evaluations = await evaluate_questions(questions, model)

        # Display results
        click.echo(f"\n{'=' * 50}")
        click.echo("EVALUATION SUMMARY")
        click.echo(f"{'=' * 50}")

        total_correctness = 0
        total_relevance = 0
        total_faithfulness = 0
        count = len(evaluations)

        for i, eval_data in enumerate(evaluations, 1):
            eval_scores = eval_data["evaluation"]
            correctness = eval_scores["correctness"]["score"]
            relevance = eval_scores["relevance"]["score"]
            faithfulness = eval_scores["faithfulness"]["score"]

            total_correctness += correctness
            total_relevance += relevance
            total_faithfulness += faithfulness

            click.echo(f"\nQ{i}: {eval_data['question'][:50]}...")
            click.echo(f"  Correctness: {correctness}/5, Relevance: {relevance}/5, Faithfulness: {faithfulness}/5")
            click.echo(f"  Answer: {eval_data['answer'][:100]}...")

        if count > 0:
            avg_correctness = total_correctness / count
            avg_relevance = total_relevance / count
            avg_faithfulness = total_faithfulness / count

            click.echo("\nAverage Scores:")
            click.echo(f"  Correctness: {avg_correctness:.2f}/5")
            click.echo(f"  Relevance: {avg_relevance:.2f}/5")
            click.echo(f"  Faithfulness: {avg_faithfulness:.2f}/5")

    asyncio.run(_evaluate())


if __name__ == "__main__":
    cli()
