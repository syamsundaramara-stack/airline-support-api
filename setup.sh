#!/bin/bash
echo "Installing all dependencies..."
pip install --upgrade pip --quiet
pip install --quiet \
  "numpy==1.26.4" "packaging==24.2" "click>=8.0.0" \
  "starlette>=0.40.0" "fastapi>=0.115.5" "uvicorn[standard]" \
  "python-dotenv==1.0.1" "pydantic-core==2.23.4" \
  "annotated-types>=0.6.0" "pydantic==2.9.2" "requests==2.32.3" \
  "sniffio" "distro>=1.7.0,<2" "jiter>=0.4.0,<1" "tqdm>4" \
  "openai>=1.54.0,<2.0.0" "tiktoken>=0.7,<1" \
  "tenacity>=8.1.0,<10,!=8.4.0" "aiohttp>=3.8.3,<4.0.0" \
  "jsonpatch>=1.33,<2.0" "langsmith>=0.1.125,<0.2.0" \
  "langchain-text-splitters>=0.3.0,<0.4.0" "langchain-core==0.3.17" \
  "SQLAlchemy>=1.4,<2.0.36" "langchain==0.3.7" "langchain-openai==0.2.6" \
  "dataclasses-json>=0.5.7,<0.7" "httpx-sse>=0.4.0,<0.5.0" \
  "pydantic-settings>=2.4.0,<3.0.0" "langchain-community==0.3.7" \
  "langchain-experimental==0.3.3" "huggingface-hub>=0.23.0" \
  "tokenizers>=0.19.1" "transformers>=4.39.0" \
  "sentence-transformers==2.7.0" "langchain-huggingface==0.1.2" \
  "pinecone-plugin-interface>=0.0.7,<0.0.8" \
  "pinecone-plugin-inference>=1.1.0,<2.0.0" "pinecone==5.3.1" \
  "pinecone-client>=5.0.0,<6.0.0" "langchain-pinecone==0.2.0" \
  "langgraph-checkpoint>=2.0.0,<3.0.0" "langgraph-sdk>=0.1.32,<0.2.0" \
  "langgraph==0.2.38" "psycopg2-binary==2.9.10" "pandas==2.2.3" \
  "pymupdf==1.24.13" "pypdf==5.1.0" "streamlit"
echo "All packages installed!"
