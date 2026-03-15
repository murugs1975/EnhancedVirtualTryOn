"""Application configuration via environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOCG_MODEL_PATH: str = "models/tocg.onnx"
    GEN_MODEL_PATH: str = "models/gen.onnx"
    POSE_MODEL_PATH: str = "models/pose_landmarker_heavy.task"
    SCHP_MODEL_PATH: str = "models/schp_lip.onnx"
    UPLOAD_DIR: str = "/tmp/tryon/uploads"
    OUTPUT_DIR: str = "/tmp/tryon/outputs"
    FINE_WIDTH: int = 768
    FINE_HEIGHT: int = 1024
    LOW_WIDTH: int = 192
    LOW_HEIGHT: int = 256
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
