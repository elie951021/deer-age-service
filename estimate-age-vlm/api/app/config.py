from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"
    session_logging_enabled: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
