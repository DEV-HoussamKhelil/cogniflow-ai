from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = "your_placeholder_key"
    environment: str = "production"
    
    class Config:
        env_file = ".env"

settings = Settings()
