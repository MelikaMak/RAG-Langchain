# run_bm25_retriever.py

import json
import os

# ✅ Load query + document data
with open("Dataset/generated_documents_type4.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

with open("Dataset/extracted_queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# ✅ Convert to LangChain Document format
documents = []
for doc in raw_docs:
    page_content = doc.get("page_content", "")
    metadata = doc.get("metadata", {})
    documents.append(Document(page_content=page_content, metadata=metadata))

# ✅ Optional: Split long docs into smaller chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# ✅ Build BM25 retriever
bm25 = BM25Retriever.from_documents(split_docs)
bm25.k = 3  # Number of results to retrieve

# ✅ Run BM25 retrieval
retrieval_results = []

for query in queries:
    results = bm25.invoke(query)  # same as get_relevant_documents

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
output_path = "Dataset/retrieval_output_bm25.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

print(f"✅ BM25 retrieval results saved to '{output_path}'")
