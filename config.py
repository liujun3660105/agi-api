from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_api_base: str
    milvus_host: str
    milvus_port: str
    milvus_username: str
    milvus_password: str

    class Config:
        env_file = ".env"  # 指定环境变量文件名
        
        
settings = Settings()