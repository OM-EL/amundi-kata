import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import base64
from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from io import BytesIO
import torch
from datetime import datetime

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory to save uploaded documents
upload_dir = "./doc"

# Choose device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

st.set_page_config(layout="wide")
st.title("Colpali-Based Multimodal RAG App")

# Ensure session state is initialized for 'indexed_files'
if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = {}

###############################################################################
# Sidebar for model selection, file upload
###############################################################################
with st.sidebar:
    st.header("Configuration Options")
    
    colpali_model = st.selectbox(
        "Select Colpali Model",
        ["vidore/colpali", "vidore/colpali-v1.2", "vidore/colqwen2-v0.1"]
    )
    multi_model_llm = st.selectbox(
        "Select Multi-Model LLM",
        ["gpt-4o", "Qwin", "Llama3.2"]
    )
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

###############################################################################
# Cache the RAG model so it only loads once
###############################################################################
@st.cache_resource
def load_rag_model(model_name: str):
    return RAGMultiModalModel.from_pretrained(model_name, verbose=10, device=device)

RAG = load_rag_model(colpali_model)

###############################################################################
# Function to create an index for a newly uploaded PDF
###############################################################################
def create_rag_index(file_path: str, file_name: str) -> str:
    # Generate a unique suffix to ensure no overlap in index_name
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_name = f"idx_{file_name}_{suffix}"

    # Index the file. We do not pass 'doc_id' since your version doesn't support it.
    RAG.index(
        input_path=file_path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=False
    )
    return index_name

###############################################################################
# 1. Upload and Index Step (only once per new file)
###############################################################################
st.subheader("Step 1: Upload & Index Your Document")

# If a file is uploaded, handle the indexing process ONCE
if uploaded_file is not None:
    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved: {uploaded_file.name}")

    # If not already indexed, create a new index
    if uploaded_file.name not in st.session_state["indexed_files"]:
        try:
            new_index_name = create_rag_index(save_path, uploaded_file.name)
            st.session_state["indexed_files"][uploaded_file.name] = new_index_name
            st.success(f"Index created for '{uploaded_file.name}': {new_index_name}")
        except ValueError as err:
            st.error(f"Error creating index: {err}")
    else:
        st.warning(f"'{uploaded_file.name}' is already indexed.")

###############################################################################
# 2. Query Step: pick an indexed document and run queries
###############################################################################
st.subheader("Step 2: Query an Indexed Document")

# If we have any indexed documents, let the user pick one
if st.session_state["indexed_files"]:
    all_docs = list(st.session_state["indexed_files"].keys())
    selected_doc = st.selectbox("Choose a Document to Query:", all_docs)
    
    user_query = st.text_input("Enter your text query:")
    
    if st.button("Search and Extract Text"):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            # Retrieve the index name for the selected document
            index_name_for_query = st.session_state["indexed_files"][selected_doc]
            
            try:
                # Perform the search
                results = RAG.search(
                    user_query,
                    k=1,
                    return_base64_results=True
                )
                if not results:
                    st.warning("No results found.")
                else:
                    # Decode base64 into an image
                    image_data = base64.b64decode(results[0].base64)
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Result Image", use_column_width=True)

                    # Use OpenAI Chat to interpret or expand upon the search result
                    try:
                        response = client.chat.completions.create(
                            model=multi_model_llm,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": user_query + "  You are an expert at identifying and extracting ESG-related information from text , answer only to the question do not give more information than asked , "},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{results[0].base64}"
                                            }
                                        },
                                    ],
                                }
                            ],
                            max_tokens=300,
                        )
                        answer = response.choices[0].message.content
                        st.subheader("LLM Response:")
                        st.markdown(answer, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error calling LLM API: {e}")
            except Exception as e:
                st.error(f"Search error: {e}")
else:
    st.info("No documents indexed yet. Please upload a PDF above.")