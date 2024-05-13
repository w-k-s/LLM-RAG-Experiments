import os
import json
import glob
from pathlib import Path
from getpass import getpass
import gdown  # download from google drive.
from haystack.components.writers import DocumentWriter
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator


# In this tutorial, we
#   1. Convert documents of different mime-types into a single, clean (no whitespace) Haystack document.
#   2. We split this document into chunks and embed those chunks in to a document store.
#   3. Ask questions of the document.
#
# In lesson 1, the data was pre-processed so we didn't use a DocumentCleaner() or DocumentSplitter().

try:

    url = "https://drive.google.com/drive/folders/1n9yqq5Gl_HWfND5bTlrCwAOycMDt5EMj"
    output_dir = "dataset/recipe_files"

    print("Downloading files")
    gdown.download_folder(url, quiet=True, output=output_dir)

    document_store = InMemoryDocumentStore()
    # Routes documents of certain mime types to the appropriate converter.
    file_type_router = FileTypeRouter(
        mime_types=["text/plain", "application/pdf", "text/markdown"]
    )
    # TextFileToDocument converts TextFiles to Haystack documents.
    text_file_converter = TextFileToDocument()
    # MarkdownToDocument converts Markdown Documents to Haystack documents
    markdown_converter = MarkdownToDocument()
    # PyPDFToDocument converts PDF documents to Haystack documents
    pdf_converter = PyPDFToDocument()
    # Merge the documents into a single list of documents
    document_joiner = DocumentJoiner()
    # Removes excessive whitespace from the documents
    document_cleaner = DocumentCleaner()
    # Splits the documents into chunks of 150 words. There is a bit of overlap to maintain context between chunks.
    document_splitter = DocumentSplitter(
        split_by="word", split_length=150, split_overlap=50
    )

    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(
        instance=file_type_router, name="file_type_router"
    )
    preprocessing_pipeline.add_component(
        instance=text_file_converter, name="text_file_converter"
    )
    preprocessing_pipeline.add_component(
        instance=markdown_converter, name="markdown_converter"
    )
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(
        instance=document_joiner, name="document_joiner"
    )
    preprocessing_pipeline.add_component(
        instance=document_cleaner, name="document_cleaner"
    )
    preprocessing_pipeline.add_component(
        instance=document_splitter, name="document_splitter"
    )
    preprocessing_pipeline.add_component(
        instance=document_embedder, name="document_embedder"
    )
    preprocessing_pipeline.add_component(
        instance=document_writer, name="document_writer"
    )

    # Route documents with mime-type text/plain to text_file_converter.
    preprocessing_pipeline.connect(
        "file_type_router.text/plain", "text_file_converter.sources"
    )
    # Route documents with mime-type application/pdf to pypdf_converter.
    preprocessing_pipeline.connect(
        "file_type_router.application/pdf", "pypdf_converter.sources"
    )
    # Route documents with mime-type text/markdown to markdown_converter.
    preprocessing_pipeline.connect(
        "file_type_router.text/markdown", "markdown_converter.sources"
    )

    # Pass the haystack documents from file converters to the document joiner.
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")

    # Clean the list of haystack documents returned from the document joiner
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")

    # Split the documents into chunks
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")

    # Embed the chunks as vectore
    preprocessing_pipeline.connect("document_splitter", "document_embedder")

    # Write the vectors the database.
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    print("Processing Documents")
    preprocessing_pipeline.run(
        {"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}}
    )

    # BEGIN QUERYING THE VECTOR DOCUMENT STORE

    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = getpass("Enter Hugging Face token:")

    template = """
    Answer the questions based on the given context.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ question }}
    Answer:
    """

    pipe = Pipeline()
    pipe.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    pipe.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=document_store)
    )
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component(
        "llm",
        HuggingFaceAPIGenerator(
            api_type="serverless_inference_api",
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
        ),
    )

    # Connect the embedder to the document embeddings
    pipe.connect("embedder.embedding", "retriever.query_embedding")

    # Connect the retriever to the prompt embedding
    pipe.connect("retriever", "prompt_builder.documents")

    # Connect prompt builder to an llm (that it can ask the prompt to)
    pipe.connect("prompt_builder", "llm")

    question = "Which of the following recipes would be met with the most disapproval from Ron Swanson? Explain why in Ron Swanson's style of speech"

    print(f"\n\nAsking the Question `{question}`")
    response = pipe.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"question": question},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        }
    )

    reply = response["llm"]["replies"][0]
    print(f"The answer is '{reply}'")
except Exception as e:
    raise e
finally:
    # Can we delete the dataset or something?
    pass
