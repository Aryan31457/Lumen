from functools import lru_cache
from langchain.chat_models import init_chat_model
from .config import settings


@lru_cache
def face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640)); return app


@lru_cache
def clip_bundle():
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-B-32"), device


def clip_image_vec(image):
    import torch
    model, preprocess, _, device = clip_bundle()
    with torch.no_grad():
        value = model.encode_image(preprocess(image).unsqueeze(0).to(device)); value /= value.norm(dim=-1, keepdim=True)
    return value[0].cpu().numpy().astype(float).tolist()


def clip_text_vec(text):
    import torch
    model, _, tokenizer, device = clip_bundle()
    with torch.no_grad():
        value = model.encode_text(tokenizer([text]).to(device)); value /= value.norm(dim=-1, keepdim=True)
    return value[0].cpu().numpy().astype(float).tolist()


@lru_cache
def text_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_doc(text): return text_embedder().encode(text, normalize_embeddings=True).tolist()
def embed_query(text): return text_embedder().encode("Represent this sentence for searching relevant passages: " + text, normalize_embeddings=True).tolist()


@lru_cache
def get_llm(): return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, temperature=0)
@lru_cache
def get_vlm(): return init_chat_model(settings.vlm_model, model_provider=settings.vlm_provider, temperature=0)
