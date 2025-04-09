"""Formula 1 Data Loader.

This module loads Formula 1 data from remote sources into a SQLite database.
"""

from io import StringIO
import logging
import os
from typing import Dict, Optional

import pandas as pd
import requests
import urllib3
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# S3 URI for F1 data
s3_uri = "https://agno-public.s3.amazonaws.com/f1"

# List of files and their corresponding table names
files_to_tables = {
    f"{s3_uri}/constructors_championship_1958_2020.csv": "constructors_championship",
    f"{s3_uri}/drivers_championship_1950_2020.csv": "drivers_championship",
    f"{s3_uri}/fastest_laps_1950_to_2020.csv": "fastest_laps",
    f"{s3_uri}/race_results_1950_to_2020.csv": "race_results",
    f"{s3_uri}/race_wins_1950_to_2020.csv": "race_wins",
}

def load_f1_data(db_path: Optional[str] = None) -> None:
    """Load Formula 1 data into a SQLite database.
    
    Downloads F1 data from S3 and loads it into tables in a SQLite database.
    
    Args:
        db_path: Optional path to the database file. If not provided, 
                 defaults to 'f1_data.db' in the current directory.
    """
    # Set default database path if not provided
    if db_path is None:
        db_path = "f1_data.db"
    
    # Database connection string
    db_url = f"sqlite:///{db_path}"
    
    logger.info(f"Loading database to {db_path}")
    engine = create_engine(db_url)

    # Load each CSV file into the corresponding SQLite table
    for file_path, table_name in files_to_tables.items():
        logger.info(f"Loading {file_path} into {table_name} table")

        try:
            # Download the file using requests
            response = requests.get(file_path, verify=False)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Read the CSV data from the response content
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)

            df.to_sql(table_name, engine, if_exists="replace", index=False)
            logger.info(f"Successfully loaded {len(df)} rows into {table_name} table")
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")

    logger.info(f"Database loaded to {db_path}")

if __name__ == "__main__":
    # Disable SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Get database path from command line argument if provided
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    load_f1_data(db_path) 