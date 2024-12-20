import os
from typing import List, Optional

import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from langchain_openai import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from httpx import Client

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

llm = AzureChatOpenAI(
    http_client=Client(),
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

# from langchain.output_parsers import OutputParser
def is_coherent(unit_truth, unit_result):
    prompt = PromptTemplate(
        input_variables=["unit_truth", "unit_result"],
        template="Are the units '{unit_truth}' and '{unit_result}' coherent in terms of metric conversion? give the scale.",
    )

    class scale(BaseModel):
        """Return a numeric answer: if coherent, return the scale factor as unit_truth/unit_result. If not coherent, return 0."""

        is_coherent: Optional[str] = Field(
            ["yes", "no"],
            description="Are the two unites coherent in terms of metric conversion. yes or no answer. f no unit_truth provided, return yes,"
        )
        scale: Optional[float] = Field(
            description="""Return a numeric answer: if coherent, return the scale factor as unit_truth/unit_result.
            Exemple if unit_truth and unit_result are the same the scale would be 1.
            Exemple if unit_truth = km and unit_result = m, the scale would be 1000. 
            If not coherent, return 0.
            If no unit_truth provided, return 1
            """,
        )

    chain = prompt | llm.with_structured_output(
        schema=scale,
        include_raw=False,
    )
    response = chain.invoke({"unit_truth": unit_truth, "unit_result": unit_result})
    return response
    # float(parser.parse(response.content))
    
    
def results_comparative_table(truth_df, result_df, with_metric_chain):
    merged_df = truth_df.merge(
        result_df, on=["company_name"], suffixes=("_truth", "_result")
    )
    if with_metric_chain:
        merged_df["unit_coherence"] = merged_df.apply(
            lambda x: is_coherent(x["unit_truth"], x["unit_result"]), axis=1
        )
        merged_df["unit_scale"] = merged_df["unit_coherence"].apply(lambda x: x.scale)
        merged_df["is_unit_coherent"] = merged_df["unit_coherence"].apply(
            lambda x: 1 if x.is_coherent == "yes" else 0
        )
    else:
        merged_df["unit_coherence"] = ''
        merged_df["unit_scale"] = 1
        merged_df["is_unit_coherent"] = 1
        
    merged_df["value_result_scale"] = merged_df.apply(
        lambda x: x["value_result"] / x["unit_scale"] if x["unit_scale"] != 0 else 0,
        axis=1,
    )
    merged_df["page_match"] = merged_df.apply(
        lambda x: abs(x["page_truth"] - x["page_result"]) <= 2, axis=1
    )
    merged_df["value_match"] = merged_df.apply(
        lambda x: 1 if x["value_truth"] == x["value_result_scale"] else 0, axis=1
    )
    merged_df["true_positive"] = merged_df.apply(
            lambda x: 1 if ((x["value_result_scale"] != 0)
        & (x["value_truth"] == x["value_result_scale"])) else 0, axis=1
        )
    merged_df["false_positive"] = merged_df.apply(
            lambda x: 1 if ((x["value_result_scale"] != 0)
        & (x["value_truth"] != x["value_result_scale"])) else 0, axis=1
        )
    merged_df["true_negative"] = merged_df.apply(
            lambda x: 1 if ((x["value_result_scale"] == 0) & (pd.isna(x["value_truth"]))) else 0, axis=1
        )
    merged_df["false_negative"] = merged_df.apply(
            lambda x: 1 if ((x["value_result_scale"] == 0) & ~(pd.isna(x["value_truth"]))) else 0, axis=1
        )

    return merged_df

def calculate_accuracy_for_sources(merged_df):
    """
    Calculate accuracy for source identification, considering tolerance of +/- 2 pages.
    """
    accuracy = merged_df["page_match"].mean()
    print(f"Accuracy for Source Identification: {accuracy:.2f}")
    return accuracy

def calculate_accuracy_for_coherent_units(merged_df):
    """
    Calculate accuracy for unit coherence.
    """
    accuracy = merged_df["is_unit_coherent"].mean()
    print(f"Accuracy for Coherent Units: {accuracy:.2f}")
    return accuracy


def calculate_precision_recall(merged_df):
    """
    Calculate precision and recall for extraction.
    """
    true_positive = (
        (merged_df["value_result_scale"] != 0)
        & (merged_df["value_truth"] == merged_df["value_result_scale"])
    ).sum()
    false_positive = (
        (merged_df["value_result_scale"] != 0)
        & (merged_df["value_truth"] != merged_df["value_result_scale"])
    ).sum()
    true_negative = (
        (merged_df["value_result_scale"] == 0) & (merged_df["value_truth"].isna())
    ).sum()
    false_negative = (
        (merged_df["value_result_scale"] == 0) & ~(merged_df["value_truth"].isna())
    ).sum()
    
    accuracy = (
        (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        if (true_positive + true_negative + false_positive + false_negative) > 0
        else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"True positives: {true_positive:.2f}")
    print(f"False positives: {false_positive:.2f}")
    print(f"True negatives: {true_negative:.2f}")
    print(f"False negatives: {false_negative:.2f}")

    return precision, recall, f1, accuracy, true_positive, false_positive, true_negative, false_negative

