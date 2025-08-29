from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE: str = "http://localhost:11434"   # Docker 默认
    OLLAMA_MODEL: str = "ollama_chat/qwen3:14b"               # 你的模型 tag
    EMBED_BASE:str= "http://localhost:11434"
    API_KEY: str=""
    OUTPUT_DIR: Path = Path("outputs")
    ENT_DIR: Path = OUTPUT_DIR / "entities"
    REL_DIR: Path = OUTPUT_DIR / "relations"


    class Config:
        env_prefix = "KG_"         # 支持环境变量覆盖
        case_sensitive = False

settings = Settings()
settings.ENT_DIR.mkdir(parents=True, exist_ok=True)
settings.REL_DIR.mkdir(parents=True, exist_ok=True)
