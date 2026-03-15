from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-agent-pipeline"
    app_env: str = "development"
    ollama_base_url: str = "http://localhost:11434"
    chroma_persist_dir: str = "data/chroma"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
