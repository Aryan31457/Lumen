from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://photos:photos@localhost:5432/photos"
    qdrant_url: str = "http://localhost:6333"
    image_dir: str = "data/images"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    vlm_provider: str = "openai"
    vlm_model: str = "gpt-4o-mini"
    min_det_score: float = 0.6
    min_face_px: int = 40
    min_blur_var: float = 40.0
    assign_threshold: float = 0.45
    centroid_match_threshold: float = 0.55
    min_cluster_size: int = 3
    caption_retries: int = 2
    max_query_attempts: int = 2
    min_good_results: int = 3


settings = Settings()
