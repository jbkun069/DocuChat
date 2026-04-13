
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    EMBEDDING_MODEL = "models/text-embedding-004"
    LLM_MODEL = "gemini-2.5-flash"

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DB_DIR = BASE_DIR / "vector_db"

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("CRITICAL: GEMINI_API_KEY not found in .env file.")