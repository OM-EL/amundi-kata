--------------------------------------------------------------------------------
                  Colpali-Based Multimodal RAG Application
--------------------------------------------------------------------------------

A professional, streamlined setup for running a Colpali-based Retrieval-Augmented Generation (RAG) application on your local machine. This application indexes PDF documents, stores them for quick access, and uses an LLM (for example, OpenAI) to answer queries about the indexed content.

![Descriptive Alt Text](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)


--------------------------------------------------------------------------------
1. REPOSITORY OVERVIEW
--------------------------------------------------------------------------------

- **app.py**  
  Main Streamlit application script.

- **doc/**  
  Folder where uploaded or existing PDF documents are stored for indexing.


- **requirements.txt**  
  Dependencies needed for the project.


--------------------------------------------------------------------------------
2. PREREQUISITES
--------------------------------------------------------------------------------

- Python **3.8+**  
- **pip** (Included by default with most Python installations)  
- **OpenAI API Key** (Required for LLM functionality)


--------------------------------------------------------------------------------
3. ENVIRONMENT CONFIGURATION
--------------------------------------------------------------------------------

Create a **.env** file in the project’s root directory with at least:

OPENAI_API_KEY=your-openai-api-key

> **Note**  
> - Keep your `.env` file out of version control if your repository is public.  
> - Add any additional environment variables as needed.


--------------------------------------------------------------------------------
4. VIRTUAL ENVIRONMENT SETUP
--------------------------------------------------------------------------------

It is highly recommended to use a virtual environment to avoid conflicts with other Python projects:

1. **Create** a virtual environment (named `venv` here):

python -m venv venv

2. **Activate** the virtual environment:
- **macOS/Linux**  
  ```
  source venv/bin/activate
  ```
- **Windows (PowerShell)**  
  ```
  .\venv\Scripts\activate
  ```


--------------------------------------------------------------------------------
5. INSTALLATION
--------------------------------------------------------------------------------

With your **virtual environment activated**:

1. **Update pip** (optional but recommended):

pip install –upgrade pip

2. **Install required packages**:

pip install -r requirements.txt

Your **requirements.txt** might include:

streamlit
byaldi          # or colpali, depending on your library version
openai
python-dotenv
torch
pillow
requests
…

--------------------------------------------------------------------------------
6. PROJECT STRUCTURE & DOCUMENT HANDLING
--------------------------------------------------------------------------------

1. **PDF Storage**  
   - Ensure there is a `doc/` folder in the project root.  
   - When you upload a PDF in the Streamlit app, it will be saved automatically into `doc/`.

2. **Indexing**  
   - The application automatically indexes newly uploaded PDFs.  
   - If you have PDFs in `doc/` before running the app, adapt indexing logic in `app.py` if you want them indexed as well.

3. **Selecting a Document**  
   - In the Streamlit UI, you will see a dropdown listing all indexed PDFs.  
   - Choose the document you want to query, then ask your question.


--------------------------------------------------------------------------------
7. RUNNING THE APPLICATION
--------------------------------------------------------------------------------

1. **Activate** your virtual environment (if not already active).  
2. **Launch Streamlit**:

streamlit run app.py


3. **Open the URL** (for example, `http://localhost:8501`) displayed in the console.


--------------------------------------------------------------------------------
8. USAGE INSTRUCTIONS
--------------------------------------------------------------------------------

1. **Upload a PDF File**  
- Use the **“Upload a PDF Document”** button in the sidebar.  
- The file is saved in `doc/` and indexed automatically.

2. **Select & Query**  
- In **Step 2** of the app, select an indexed PDF from the dropdown.  
- Enter your query text, then click **“Search and Extract Text.”**  
- The RAG search retrieves a relevant page as an image, which is passed to the LLM for processing.

3. **LLM Response**  
- The OpenAI API key in `.env` is used for authenticating with the model.  
- The returned text is displayed under **“LLM Response.”**


--------------------------------------------------------------------------------
9. TROUBLESHOOTING
--------------------------------------------------------------------------------

- **Missing or Invalid OpenAI Key**  
Verify `OPENAI_API_KEY` is set in your `.env`. Ensure your key is valid and that you have adequate usage limits.

- **Indexing Issues**  
Confirm the `doc/` folder exists and that the PDF is not locked or corrupted. Re-indexing the same PDF might cause conflicts if your library version does not handle duplicates gracefully.

- **Dependency Conflicts**  
Confirm all dependencies are installed using `pip install -r requirements.txt`. Make sure your Python version matches what your libraries require.


--------------------------------------------------------------------------------
10. ADDITIONAL NOTES
--------------------------------------------------------------------------------

- **GPU Usage**  
If you have a CUDA-capable GPU, verify that you installed the correct PyTorch version. Otherwise, the application runs on CPU (or Apple MPS if available).

- **Stopping the App**  
Press **CTRL + C** in the terminal or close the terminal to stop the local server.

