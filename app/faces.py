import cv2
from qdrant_client import models
from .config import settings
from .models import face_app
from .qdrant_store import client


def _blur_var(image, bbox):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]; crop = image[y1:y2, x1:x2]
    return 0.0 if crop.size == 0 else float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def detect(path):
    image = cv2.imread(path)
    if image is None: raise ValueError(f"Cannot read image: {path}")
    found = []
    for face in face_app().get(image):
        bbox = [float(v) for v in face.bbox]; blur = _blur_var(image, bbox)
        found.append({"bbox": bbox, "det_score": float(face.det_score), "blur_var": blur,
                      "quality_ok": float(face.det_score) >= settings.min_det_score and min(bbox[2]-bbox[0], bbox[3]-bbox[1]) >= settings.min_face_px and blur >= settings.min_blur_var,
                      "embedding": face.normed_embedding.astype(float).tolist()})
    return image, sorted(found, key=lambda item: item["bbox"][0])


def try_assign(vector):
    points = client().query_points("faces", query=vector, limit=7, with_payload=True,
        query_filter=models.Filter(must=[models.FieldCondition(key="person_id", range=models.Range(gt=0))])).points
    if not points or points[0].score < settings.assign_threshold: return None, points[0].score if points else 0.0
    votes = {}; penalties = {}; verified_matches = {}
    for point in points:
        payload = point.payload or {}
        pid = payload.get("person_id")
        image_id = payload.get("image_id")
        if pid is not None and image_id is not None and point.score >= settings.assign_threshold:
            weight = settings.identity_feedback_positive_weight if payload.get("human_verified") else 1.0
            weighted_score = point.score * weight
            votes.setdefault(pid, {})[image_id] = max(weighted_score, votes.get(pid, {}).get(image_id, 0.0))
            if payload.get("human_verified"):
                verified_matches.setdefault(pid, set()).add(image_id)
        if image_id is not None and point.score >= settings.assign_threshold * 0.8:
            for wrong_pid in payload.get("forbidden_person_ids", []):
                penalties.setdefault(wrong_pid, {})[image_id] = max(
                    point.score, penalties.get(wrong_pid, {}).get(image_id, 0.0),
                )
    # A single near-duplicate or bad embedding must not assign a new face.
    ranked = sorted((
        (sum(scores.values()) - settings.identity_feedback_penalty_weight * sum(penalties.get(pid, {}).values()), pid, scores)
        for pid, scores in votes.items()
    ), reverse=True)
    if not ranked:
        return None, points[0].score
    best_score, best_pid, best_images = ranked[0]
    if best_score <= 0:
        return None, points[0].score
    if len(best_images) < 2 and not verified_matches.get(best_pid):
        return None, points[0].score
    if len(best_images) < 2 and len(ranked) > 1 and ranked[1][0] >= best_score / 1.25:
        return None, points[0].score
    if len(ranked) > 1:
        second_score, _, second_images = ranked[1]
        if len(second_images) >= 2 and best_score < second_score * 1.25:
            return None, points[0].score
    return best_pid, points[0].score
