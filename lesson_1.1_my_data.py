import os
import structlog
import csv
import json
from uuid import uuid4
from getpass import getpass
from haystack import Document
from haystack import Pipeline
from datasets import load_dataset
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator


try:
    # A DocumentStore stores the Documents that the question answering system uses to find answers to your questions
    print("Initializing In-memory document store")
    document_store = InMemoryDocumentStore()

    # Load Data set. The data has already been cleaned and split into tokens
    print("Creating haystack documents from training data")
    docs = []
    with open("dataset/expenses.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            docs.append(
                Document(
                    content=json.dumps(row), meta={"date": row["Date"], "id": uuid4()}
                )
            )

    # Embedders in Haystack transform texts or Documents into vector representations using pre-trained models.
    # You can then use the embeddings in your pipeline for tasks like question answering, information retrieval, and more.
    print("Initisialsing document embedder")
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    # all warm_up() to download the embedding model.
    doc_embedder.warm_up()

    print("Embedding documents")
    docs_with_embeddings = doc_embedder.run(docs)

    print("Writing embeddings to document store")
    document_store.write_documents(docs_with_embeddings["documents"])

    # RAG PIPELINE

    # Previously, we created an embeddign for the documents that the question will be answered from
    # Now we're going to create an embedding for the actual question being asked.
    # Same model should be used for embedding the document as embedding the query.
    print("Initisialsing text embedder")
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize a InMemoryEmbeddingRetriever and make it use the InMemoryDocumentStore you initialized earlier in this tutorial.
    # This Retriever will get the relevant documents to the query.
    print("Initisialsing in-memory retriever for embeddings in document store")
    retriever = InMemoryEmbeddingRetriever(document_store)

    # Create a custom prompt for a generative question answering task using the RAG approach.
    # The prompt should take in two parameters: documents, which are retrieved from a document store, and a question from the user.
    # Use the Jinja2 looping syntax to combine the content of the retrieved documents in the prompt
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    print("Looking for OpenAI API Key")
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
    generator = OpenAIGenerator(model="gpt-3.5-turbo")

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    # Question: Why do we need these connections? To do: make a diagram to make sense of this part
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    while True:
        question = input("Ask your question. Press enter to submut")
        if question == "quit":
            break

        print("Asking the question")
        response = basic_rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )

        reply = response["llm"]["replies"][0]
        print(f"The answer is '{reply}'")
    print("Done")
except Exception as e:
    raise e
finally:
    # Can we delete the dataset or something?
    pass
