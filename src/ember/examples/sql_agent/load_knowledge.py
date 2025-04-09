import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
CWD = Path(__file__).parent
KNOWLEDGE_DIR = CWD.joinpath("knowledge")

def load_knowledge():
    """Load the knowledge base and print a summary."""
    logger.info("Loading SQL agent knowledge.")
    
    # Load table metadata
    tables = {}
    for file_path in KNOWLEDGE_DIR.glob("*.json"):
        with open(file_path, "r") as f:
            table_data = json.load(f)
            table_name = table_data.get("table_name")
            if table_name:
                tables[table_name] = table_data
    
    logger.info(f"Loaded metadata for {len(tables)} tables.")
    
    # Load sample queries
    sample_queries_path = KNOWLEDGE_DIR / "sample_queries.sql"
    sample_queries = []
    
    if sample_queries_path.exists():
        with open(sample_queries_path, "r") as f:
            content = f.read()
            
            # Parse the sample queries
            query_blocks = []
            current_block = {"description": "", "query": ""}
            in_description = False
            in_query = False
            
            for line in content.split("\n"):
                if line.strip() == "-- <query description>":
                    in_description = True
                    in_query = False
                    if current_block["query"]:
                        query_blocks.append(current_block)
                        current_block = {"description": "", "query": ""}
                elif line.strip() == "-- </query description>":
                    in_description = False
                elif line.strip() == "-- <query>":
                    in_query = True
                    in_description = False
                elif line.strip() == "-- </query>":
                    in_query = False
                elif in_description:
                    current_block["description"] += line.replace("-- ", "") + "\n"
                elif in_query:
                    current_block["query"] += line + "\n"
            
            if current_block["query"]:
                query_blocks.append(current_block)
            
            sample_queries = query_blocks
    
    logger.info(f"Loaded {len(sample_queries)} sample queries.")
    
    # Print a summary of the knowledge base
    logger.info("Knowledge base summary:")
    logger.info(f"Tables: {', '.join(tables.keys())}")
    logger.info(f"Sample queries: {len(sample_queries)}")
    
    logger.info("SQL agent knowledge loaded.")
    
    return {
        "tables": tables,
        "sample_queries": sample_queries
    }

if __name__ == "__main__":
    load_knowledge()
