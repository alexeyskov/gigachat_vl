from os import environ
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr


load_dotenv(".env")


class Settings(BaseModel):
    TEACHER_API_KEY: SecretStr
    TEACHER_BASE_URL: str
    TEACHER_MODEL: str
    TEACHER_TIMEOUT_SECONDS: float = 45.0
    HF_TOKEN: Optional[SecretStr] = None
    HF_REPO_ID: Optional[str] = None


SETTINGS = Settings(**environ)
