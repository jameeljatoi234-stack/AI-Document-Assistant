# AI Document Assistant

A simple Streamlit RAG application that lets you:

- Upload **PDF, DOCX, TXT and MD** files.
- Load supported files from a **public Google Drive file or folder link**.
- Extract document text and preserve **filename** and **PDF page number** metadata.
- Split text into overlapping chunks.
- Create **Sentence Transformers** embeddings.
- Store embeddings in a **FAISS** vector index.
- Run **semantic search + keyword search** as a hybrid search.
- Send the best retrieved chunks to **Groq**.
- Ask questions that are answered only from the supplied document context.
- See the retrieved source chunks after every answer.
- Reuse already-created embeddings during the Streamlit session.

## Project files

```text
AI-Document-Assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Create the project

Create a new GitHub repository and add:

- `app.py`
- `requirements.txt`
- `README.md`

Copy the supplied code into each file.

## 2. Add your Groq API key

The app does **not** hardcode the API key.

For local Streamlit use, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

For Streamlit Community Cloud:

1. Open your deployed app.
2. Open **Settings**.
3. Open **Secrets**.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Save the secrets and restart the app if necessary.

## 3. Install and run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Using local files

In the sidebar:

1. Upload one or more PDF, DOCX, TXT or MD files.
2. Click **Process documents**.
3. Wait for extraction, chunking and embedding.
4. Ask questions in the chat box.

The app processes the current document set once and keeps the pipeline in
`st.session_state`, so it does not recreate embeddings for every question.

## 5. Using Google Drive

Paste a Google Drive **file link** or **folder link** into the sidebar.

Important:

- The Drive item must be shared as **Anyone with the link**.
- Supported downloaded file types are PDF, DOCX, TXT and MD.
- Private Google Drive access is not included because that requires Google
  OAuth/service-account authentication.

The downloaded Drive files go through exactly the same pipeline as local uploads.

## How the app works

### Step 1 — Extraction

Separate functions handle:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

PDF extraction stores page numbers. DOCX/TXT/MD keep the filename, but page
numbers are shown as unavailable because these formats do not provide reliable
page boundaries through simple text extraction.

### Step 2 — Chunking

`create_chunks()` splits extracted text into overlapping chunks.

Default settings:

- Chunk size: `1200` characters
- Overlap: `200` characters

Each chunk keeps:

- Filename
- Page number when available
- Chunk number
- Text

### Step 3 — Embeddings

The app uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The Sentence Transformers model is loaded with `st.cache_resource`.

Chunk embeddings are created once when the document set is processed.

### Step 4 — FAISS semantic search

The app uses normalized embeddings with:

```python
faiss.IndexFlatIP
```

Because the embeddings are normalized, inner product behaves like cosine
similarity.

### Step 5 — Keyword search

The question is converted into important keywords after removing common words.

Each chunk receives a keyword score based on:

- Query-word coverage
- Small frequency bonus

### Step 6 — Hybrid search

The final score is:

```text
75% semantic score + 25% keyword score
```

The highest-ranked chunks are passed to Groq.

### Step 7 — Groq answer

The app uses:

```text
openai/gpt-oss-20b
```

The system prompt tells the model to answer only from the retrieved context and
to say:

```text
The information is not available in the provided documents.
```

when the answer cannot be supported by the supplied chunks.

## Notes

### Scanned PDFs

`pypdf` extracts embedded text. A scanned/image-only PDF may return little or no
text. OCR is intentionally not included in this simple version.

### DOCX page numbers

DOCX files do not contain simple reliable page boundaries because pagination
depends on layout, fonts and the word processor. This app therefore shows
`Page unavailable` for DOCX.

### Google Drive limitations

This beginner version uses `gdown` and public share links instead of Google
OAuth. If you later need private Drive access, add Google OAuth or a service
account as a separate feature.

## Suggested deployment

The easiest deployment path is:

1. Push the three files to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository.
4. Set the main file to `app.py`.
5. Add `GROQ_API_KEY` in Streamlit Secrets.
6. Deploy.
