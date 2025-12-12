from llama_index.core import Document
from llama_index.core.schema import TextNode
from typing import List, Dict

def process_email_data(json_data: Dict, file_path: str) -> List[Document]:
    documents = []

    for thread_id, data in json_data.items():
        # 1. Extract Global Metadata (The Parent Context)
        # This metadata will be attached to every single atomic chunk from this thread
        thread_meta = data['thread_metadata']
        base_metadata = {
            "file_path": file_path,
            "thread_id": thread_id,
            "subject": thread_meta['subject'],
            "source_type": "email",
            "participants": ", ".join(thread_meta['participants']),
            "created_at": thread_meta['latestSentDateUTC']
        }

        # --- CHUNK TYPE A: The Summary (Narrative Context) ---
        # Good for: "What happened in the Andrea Hills thread?"
        if data.get('thread_summary'):
            summary_doc = Document(
                text=data['thread_summary'],
                metadata={
                    **base_metadata,
                    "node_type": "summary",
                    "category": "narrative"
                }
            )
            documents.append(summary_doc)

        # --- CHUNK TYPE B: Knowledge Items (Fact Context) ---
        # Good for: "What is the policy on guide rooms?"
        for item in data.get('knowledge_items', []):
            # We format the text to be self-contained and descriptive
            content = (
                f"Fact Type: {item['type']}\n"
                f"Scope: {item['scope']}\n"
                f"Statement: {item['statement']}\n"
                f"Applicability: {item['applicability']}"
            )
            
            k_doc = Document(
                text=content,
                metadata={
                    **base_metadata,
                    "node_type": "knowledge_item",
                    "scope": item['scope'],  # Allows filtering by 'booking_policy'
                    "category": "fact"
                }
            )
            documents.append(k_doc)

        # --- CHUNK TYPE C: Q&A Pairs (Dialogue Context) ---
        # Good for: "How do we handle cancellation requests?"
        for qa in data.get('qa_pairs', []):
            content = (
                f"Question: {qa['q']}\n"
                f"Answer: {qa['ans']}\n"
                f"Context: {qa['context']}"
            )
            
            qa_doc = Document(
                text=content,
                metadata={
                    **base_metadata,
                    "node_type": "qa_pair",
                    "category": "dialogue"
                }
            )
            documents.append(qa_doc)
            
        # --- CHUNK TYPE D: Seasonal Knowledge (Time-Based Context) ---
        # Good for: "Is November a good time for the Annapurna trek?"
        for season_item in data.get('seasonal_knowledge', []):
            # Create a focused text representation
            content = (
                f"Seasonal Fact: {season_item['statement']}\n"
                f"Time Period: {season_item['date_or_range']}\n"
                f"Impact: {season_item['impact']}"
            )
            
            season_doc = Document(
                text=content,
                metadata={
                    **base_metadata,
                    "node_type": "seasonal_knowledge",
                    "category": "environment",
                    # CRITICAL: Add the specific month/season as a filterable field
                    "season": season_item['date_or_range'] 
                }
            )
            documents.append(season_doc)

    return documents



def get_documents_from_email() -> List[Document]:
    import os
    from pathlib import Path
    import json
    ROOT_DIR  = Path(__file__).resolve().parent
    json_file  =  os.path.join(ROOT_DIR, 'final_email_kb.json')
    with open(json_file) as file:
        data  =  json.load(file)
        return process_email_data(json_data=data, file_path=json_file)