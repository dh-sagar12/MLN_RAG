#!/bin/bash
# Quick start script for RAG Chatbot

echo "Starting RAG Chatbot..."
echo "Make sure you have:"
echo "1. PostgreSQL with pgvector installed and running"
echo "2. Created the database: rag_chatbot"
echo "3. Set up your .env file with OPENAI_API_KEY"
echo ""
echo "Starting server on http://localhost:8000"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
