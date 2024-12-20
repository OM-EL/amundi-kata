from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from httpx import Client
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd
from langchain_community.vectorstores import Chroma
import os

os.getenv("TIKTOKEN_CACHE_DIR")

ALTO_STUDIO_TOKEN = os.getenv("ALTO_STUDIO_TOKEN")
tiktoken_cache_dir = "./tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(
    os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")
)

api_base = "https://studio-llm-gateway.intramundi.com/"  # In studio
deployment_name = "gpt-4o"  # gpt-35-turbo or gpt-35-turbo-16k or gpt-4-turbo
temperature = 0



embeddings = AzureOpenAIEmbeddings(
    default_headers={},
    azure_endpoint=api_base,
    deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
    openai_api_key=ALTO_STUDIO_TOKEN,
)

llm = AzureChatOpenAI(
    # http_client=Client(),
    azure_endpoint=api_base,
    deployment_name=deployment_name,
    openai_api_key=ALTO_STUDIO_TOKEN,
    model_name="gpt-4o",
    openai_api_version="2023-03-15-preview",
    streaming=False,  # for receive the response in streaming
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],  # for display in the output intermediate steps
    verbose=True,
    temperature=temperature,
)


def extract_info(path_vectorstore, secondary_theme, description_theme, year):
    text = f"""What is {secondary_theme} in {year}: {description_theme}"""
    class ExtractionDataDetails(BaseModel):
        f"""Information about {secondary_theme}."""

        value: Optional[float] = Field(
            None,
            description=f"The reported value of {secondary_theme} in {year}. {secondary_theme} refers to {description_theme}.",
        )
        unit: Optional[str] = Field(
            None,
            description=f"The unit of the reported value of {secondary_theme} in {year}.",
        )
        page: Optional[int] = Field(
            None,
            description=f"Page of the document from which the {secondary_theme} in {year}.",
        )
        extract: Optional[str] = Field(
            None,
            description=f"The paragraph in the Page of the document from which the {secondary_theme} was extracted or the name of the table or graph it was extracted from.",
        )
        extract_type: Optional[str] = Field(
            None,
            description=f"Weither the reported value of {secondary_theme} was extracted from a text, a table or a graph.",
        )
        paragraph: Optional[str] = Field(
            None,
            description=f"The paragraph title from where the value of {secondary_theme} was extracted.",
        )


    class ExtractionData(BaseModel):
        """Extracted information about ESG data, specifically focusing on GHG emissions."""

        data_extracted: List[ExtractionDataDetails]


    # Define a custom prompt to provide instructions and any additional context.
    # 1) Add examples into the prompt template to improve extraction quality.
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at identifying and extracting ESG-related information from text."
                f"Focus on extracting details about {secondary_theme}, including values, units, page numbers, and associated paragraphs or tables. "
                "Extract nothing if no relevant information can be found in the text.",
            ),
            ("human", "{text}"),
        ]
    )

    extractor = prompt | llm.with_structured_output(
        schema=ExtractionDataDetails,#ExtractionData,
        include_raw=False,
    )
    vectorstore = Chroma(
        persist_directory=path_vectorstore, embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )  # Only extract from first document

    rag_extractor = {"text": retriever} | extractor
    result = rag_extractor.invoke(text)
    return result