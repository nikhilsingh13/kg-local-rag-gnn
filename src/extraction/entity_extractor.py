"""
Extracting structured entities and relationships from paper's text chunks.

Entity types:
    Paper     — title, arXiv ID, year
    Author    — name, affiliation
    Model     — GNN architecture name (GCN, GAT, GraphSAGE, GIN …)
    Dataset   — benchmark dataset name (Cora, Citeseer, OGB …)
    Task      — ML task (node classification, link prediction …)
    and anything else if required.

Relationship types:
    authored_by    Paper → Author
    proposes       Paper → Model
    evaluates_on   Paper → Dataset
    addresses      Paper → Task
    cites          Paper → Paper
    improves_on    Model → Model

    db strucuture.
"""

import json
import re
import logging
import yaml

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient
from tqdm.auto import tqdm

from src.config import (HF_TOKEN,
                        LLM_MODEL_ID,
                        MAX_TOKENS,
                        DATA_PROCESSED_DIR,
                        ENTITIES_PATH,
                        ENTITY_PROMPT_PATH,
                        RELATIONSHIP_PROMPT_PATH)

@dataclass
class Entity:
    type: str           # Paper | Author | Model | Dataset | Task
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    source: str         # entity name
    relation: str       # relationship type
    target: str         # entity name
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relationships: list[Relationship]
    source_chunk_id: str

class EntityExtractor:
    """Extract entities and relationships from text using an LLM."""
    
    def __init__(self,
                 model_id: str = LLM_MODEL_ID,
                 hf_token: str = HF_TOKEN,
                 ) -> None:
         """
         Args:
            model_id:  HF model ID — used via Inference API (serverless).
                Defaults to config.LLM_MODEL_ID.
            hf_token:  HuggingFace token for gated/private models.
            """
         
         self.model_id = model_id
         self.client = InferenceClient(token=hf_token)
    
    def _load_prompt(self, path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _call_llm(self, system: str, user: str) -> str:
        """Call language model for extraction"""
        messages = [
            {
                "role": "system", 
                "content": system
            },
            {
                "role": "user", 
                "content": user
            }]
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=MAX_TOKENS,
                temperature=0.1)
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"API Error: {e}. ")
            return ""
        
    def _parse_entities(self, raw: str, chunk_id: str) -> ExtractionResult:
        """Parse entities."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            entities = [Entity(**e) for e in data.get("entities", [])]
            return entities    
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Entity parser failed for chunk {chunk_id}: {e}")
            return []

    def _parse_relationships(self, raw:str, chunk_id: str) -> ExtractionResult:
        """Parse relationships."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            relationships = [Relationship(**r) for r in data.get("relationships", [])]
            return relationships
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"relationship parser failed for chunk {chunk_id}: {e}")
            return []
    
    def extract(self, text: str, chunk_id: str) -> ExtractionResult:
        
        # loading prompts
        entity_prompt = self._load_prompt(ENTITY_PROMPT_PATH)
        relationship_prompt = self._load_prompt(RELATIONSHIP_PROMPT_PATH)

        # entities
        entity_raw = self._call_llm(entity_prompt["system"], 
                                    entity_prompt["user"].format(text=text))
        entities = self._parse_entities(entity_raw, chunk_id)

        # relationships
        entity_names = [e.name for e in entities]
        rel_raw = self._call_llm(relationship_prompt["system"],
                                 relationship_prompt["user"].format(text=text, 
                                                                    entity_names=entity_names))
        relationships = self._parse_relationships(rel_raw, chunk_id)
        
        return ExtractionResult(entities=entities,
                                relationships=relationships, 
                                source_chunk_id=chunk_id)
    
    def extract_batch(self, 
                      texts: list[tuple[str, str]], 
                      batch_size: int = 5,
                      resume_from: str | None = None) -> list[ExtractionResult]:
        processed_ids = set()
        
        if resume_from and Path(resume_from).exists():
            with open(resume_from) as f:
                for line in f:
                    data = json.loads(line)
                    processed_ids.add(data["source_chunk_id"])
        
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            for text, chunk_id in batch:
                if chunk_id in processed_ids:
                    continue
                results.append(self.extract(text, chunk_id))
        return results

if __name__ == "__main__":
    chunks = [json.loads(l) for l in open(DATA_PROCESSED_DIR / "chunks.jsonl")]
    texts = [(c["text"], f"{c['paper_id']}_chunk_{c['chunk_index']}") for c in chunks]
    
    extractor = EntityExtractor()
    results = extractor.extract_batch(texts, batch_size=3, resume_from=ENTITIES_PATH)
    
    with open(ENTITIES_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    
    print(f"Saved {len(results)} results.")