# **Colpali-Based Multimodal RAG Application**

A streamlined setup for running a **Colpali-based Retrieval-Augmented Generation (RAG)** application locally. This tool indexes PDF documents, stores them for efficient retrieval, and leverages an LLM (e.g., OpenAI) to answer queries about the indexed content.

---

Demo : https://www.loom.com/embed/a03e80c4233341a5b37160e4c4e8ac68?sid=55026e02-c6b7-4842-acee-3dcdd39bb6e6

## **Table of Contents**

1. [Introduction](#introduction)  
2. [RAG Architecture](#rag-architecture)  
3. [Project Rationale and Constraints](#project-rationale-and-constraints)  
4. [Key Features](#key-features)  
5. [Repository Overview](#repository-overview)  
6. [Prerequisites](#prerequisites)  
7. [Environment Configuration](#environment-configuration)  
8. [Virtual Environment Setup](#virtual-environment-setup)  
9. [Installation](#installation)  
10. [Project Structure & Document Handling](#project-structure--document-handling)  
11. [Running the Application](#running-the-application)  
12. [Usage Instructions](#usage-instructions)  
13. [Troubleshooting](#troubleshooting)  
14. [Additional Notes](#additional-notes)

---

## **Introduction**

The **Colpali-Based Multimodal RAG Application** allows you to:

- Index and retrieve information from PDF documents.
- Query indexed content via an LLM like OpenAI.
- Visualize results with a user-friendly Streamlit interface.

---

## **RAG Architecture**

Explore the difference between classic and **Colpali-based systems**:  

![Architecture Overview](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

<img width="1243" alt="Screenshot 2024-12-28 at 15 13 15" src="https://github.com/user-attachments/assets/a6bc5d28-7cb0-4d79-af61-279f7dc42e9b" />

*Source: [Medium Article](https://medium.com/@simeon.emanuilov/colpali-revolutionizing-multimodal-document-retrieval-324eab1cf480)*

---

## **Project Rationale and Constraints**

### Goals

1. **Complex Document Support**: Process documents with mixed content (text, tables, graphs, images).
2. **Answer Transparency**: Allow users to verify answers by referencing the exact page or snippet.
3. **Performance Balance**: Indexing may take time; retrieval must be instantaneous.
4. **Standalone System**: No integration with external systems was required.

---

## **Key Features**

1. **Multimodal Processing**: Combines text and visual embeddings for better retrieval accuracy.
2. **Transparent UI**: Users can view referenced pages/snippets for validation.
3. **Performance Optimization**: Instantaneous query resolution despite computationally heavy indexing.
4. **Stand-Alone Application**: Requires no external system integration, simplifying deployment.

---

## **Repository Overview**

- `app.py`  
  The main Streamlit application script.

- `doc/`  
  Directory for storing uploaded PDF documents.

- `requirements.txt`  
  List of required dependencies for the project.

---

## **Prerequisites**

- **Python** 3.8+  
- **pip** (bundled with most Python installations)  
- **OpenAI API Key** (for LLM functionalities)

---

## **Environment Configuration**

1. Create a `.env` file in the project root:
   ```plaintext
   OPENAI_API_KEY=your-openai-api-key
   ```

2. Keep `.env` out of version control if the repository is public.

---

## **Virtual Environment Setup**

1. Create a virtual environment:  
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:  
   - macOS/Linux:  
     ```bash
     source venv/bin/activate
     ```
   - Windows (PowerShell):  
     ```bash
     .\venv\Scripts\activate
     ```

---

## **Installation**

1. Activate your virtual environment.  
2. Update pip (optional but recommended):  
   ```bash
   pip install --upgrade pip
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

Sample `requirements.txt`:
```
streamlit
colpali
openai
python-dotenv
torch
pillow
requests
```

---

## **Project Structure & Document Handling**

1. **PDF Storage**  
   - Ensure a `doc/` folder exists in the project root.  
   - Uploaded PDFs are automatically saved here.

2. **Indexing**  
   - PDFs are indexed automatically upon upload.  
   - Pre-existing PDFs in `doc/` may require adjustments in `app.py`.

3. **Selecting a Document**  
   - Use the dropdown in the Streamlit app to select indexed PDFs.  
   - Ask queries about the selected document.

---

## **Running the Application**

1. Activate the virtual environment.  
2. Run the application:  
   ```bash
   streamlit run app.py
   ```
3. Open the URL displayed in the console (e.g., `http://localhost:8501`).

---

## **Usage Instructions**

1. **Upload a PDF**  
   - Use the **“Upload a PDF Document”** button in the sidebar.  
   - The document is saved in `doc/` and indexed.

2. **Query the Document**  
   - Select a document from the dropdown.  
   - Enter a query, then click **“Search and Extract Text.”**  
   - Results include a retrieved page image and LLM-generated text.

3. **LLM Responses**  
   - LLM responses are based on the OpenAI API key in your `.env`.

---

## **Troubleshooting**

### **Common Issues**

- **Missing or Invalid OpenAI Key**  
  Ensure `OPENAI_API_KEY` is correctly set in `.env`.

- **Indexing Problems**  
  Verify `doc/` exists and that PDFs are not locked or corrupted.

- **Dependency Conflicts**  
  Reinstall dependencies with:  
  ```bash
  pip install -r requirements.txt
  ```

### **Advanced Debugging**

- Check Python and library versions.  
- Re-index PDFs manually if necessary.

---

## **Additional Notes**

- **GPU Support**  
  Use a CUDA-enabled GPU or Apple MPS for faster processing (if supported).  
  Ensure you install the appropriate PyTorch version.

- **Stopping the App**  
  Press **CTRL + C** or close the terminal to stop the server.

---

This improved structure ensures clarity and professionalism, making the README easy to read and follow for both technical and non-technical users.

