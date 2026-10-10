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
    votes = {}
    for point in points:
        if point.score >= settings.assign_threshold: votes[point.payload["person_id"]] = votes.get(point.payload["person_id"], 0) + point.score
    return max(votes, key=votes.get), points[0].score
