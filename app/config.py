from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://photos:photos@localhost:5432/photos"
    qdrant_url: str = "http://localhost:6333"
    image_dir: str = "data/images"
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    hf_token_key: str | None = None
    min_det_score: float = 0.6
    min_face_px: int = 40
    min_blur_var: float = 40.0
    assign_threshold: float = 0.60
    centroid_match_threshold: float = 0.68
    cluster_similarity_threshold: float = 0.52
    clip_search_weight: float = 0.70
    text_search_weight: float = 0.30
    lexical_search_weight: float = 0.10
    search_candidate_multiplier: int = 10
    search_min_clip_confidence: float = 0.18
    search_min_text_confidence: float = 0.55
    min_cluster_size: int = 3
    caption_max_new_tokens: int = 32


settings = Settings()
