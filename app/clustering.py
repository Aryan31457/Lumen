import numpy as np
from sklearn.cluster import AgglomerativeClustering
from qdrant_client import models
from .config import settings
from .db import conn
from .qdrant_store import client
from . import people_service as people


def run_cluster_job():
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id FROM faces WHERE person_id IS NULL AND quality_ok")
        ids = [r["face_id"] for r in cur.fetchall()]
    if len(ids) < settings.min_cluster_size:
        return {"pending": len(ids), "new_people": 0, "matched_existing": 0}

    points = client().retrieve("faces", ids=ids, with_vectors=True, with_payload=True)
    points = [p for p in points if p.vector is not None]
    if len(points) < settings.min_cluster_size:
        return {"pending": len(ids), "new_people": 0, "matched_existing": 0}
    ids = [p.id for p in points]
    vectors = np.asarray([p.vector for p in points], dtype=np.float32)
    image_ids = [p.payload.get("image_id") if p.payload else None for p in points]

    # Complete linkage requires every pair in a cluster to be similar enough;
    # unlike density chaining, one bridge face cannot join unrelated people.
    distance_threshold = 1.0 - settings.cluster_similarity_threshold
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="complete",
        distance_threshold=distance_threshold,
    ).fit_predict(vectors)
    new = matched = 0
    for label in set(labels) - {-1}:
        indexes = np.where(labels == label)[0]
        cluster = [ids[i] for i in indexes]
        distinct_images = {image_ids[i] for i in indexes if image_ids[i] is not None}
        # Repeated detections from one photo are not evidence for an identity.
        if len(distinct_images) < settings.min_cluster_size or len(distinct_images) != len(indexes):
            continue

        votes = {}; penalties = {}
        for idx in indexes:
            hits = client().query_points(
                "faces", query=vectors[idx].tolist(), limit=5, with_payload=True,
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="person_id", range=models.Range(gt=0),
                )]),
            ).points
            for hit in hits:
                if hit.score < settings.centroid_match_threshold:
                    continue
                payload = hit.payload or {}
                pid, image_id = payload.get("person_id"), payload.get("image_id")
                if pid is not None and image_id is not None:
                    if hit.score >= settings.centroid_match_threshold:
                        weight = settings.identity_feedback_positive_weight if payload.get("human_verified") else 1.0
                        votes.setdefault(pid, {}).setdefault(image_id, []).append(hit.score * weight)
                    if hit.score >= settings.centroid_match_threshold * 0.8:
                        for wrong_pid in payload.get("forbidden_person_ids", []):
                            penalties.setdefault(wrong_pid, {}).setdefault(image_id, []).append(hit.score)

        ranked = sorted(
            ((sum(max(scores) for scores in images.values()) - settings.identity_feedback_penalty_weight * sum(max(scores) for scores in penalties.get(pid, {}).values()), pid, images)
             for pid, images in votes.items()), reverse=True,
        )
        if ranked and ranked[0][0] > 0 and len(ranked[0][2]) >= 2:
            best_score, pid, best_images = ranked[0]
            if len(ranked) == 1 or len(ranked[1][2]) < 2 or best_score >= ranked[1][0] * 1.25:
                matched += 1
            else:
                pid = people.create_person()
                new += 1
        else:
            pid = people.create_person()
            new += 1
        people.assign_faces(cluster, pid, "cluster")
    return {"pending": len(ids), "new_people": new, "matched_existing": matched}
