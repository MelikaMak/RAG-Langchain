
# run_retriever.py

import json
import os

# ✅ Load API key from config.py
import config

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# ✅ Load generated documents
with open("Dataset/generated_documents_type4.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

documents = []
for doc in raw_docs:
    page_content = doc.get("page_content", "")
    metadata = doc.get("metadata", {})
    documents.append(Document(page_content=page_content, metadata=metadata))

# ✅ Optional: Split long documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# ✅ Create OpenAI embeddings
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Build FAISS index
db = FAISS.from_documents(split_docs, embedding)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Load queries
with open("Dataset/extracted_queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# ✅ Retrieve documents silently
retrieval_results = []

for query in queries:
    results = retriever.invoke(query)

    retrieved_items = [
        {
            "content": res.page_content,
            "metadata": res.metadata
        }
        for res in results
    ]

    retrieval_results.append({
        "query": query,
        "retrieved_documents": retrieved_items
    })

# ✅ Save results to file
output_path = "Dataset/retrieval_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

print(f"✅ Retrieval results saved to '{output_path}'")
