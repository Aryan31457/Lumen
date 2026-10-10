import numpy as np
from sklearn.cluster import HDBSCAN
from qdrant_client import models
from .config import settings
from .db import conn
from .qdrant_store import client
from . import people_service as people


def run_cluster_job():
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id FROM faces WHERE person_id IS NULL AND quality_ok"); ids=[r['face_id'] for r in cur.fetchall()]
    if len(ids) < settings.min_cluster_size: return {"pending":len(ids),"new_people":0,"matched_existing":0}
    points=client().retrieve("faces",ids=ids,with_vectors=True); vectors=np.asarray([p.vector for p in points],dtype=np.float32); ids=[p.id for p in points]
    labels=HDBSCAN(min_cluster_size=settings.min_cluster_size,min_samples=2).fit_predict(vectors); new=matched=0
    for label in set(labels)-{-1}:
        indexes=np.where(labels==label)[0]; cluster=[ids[i] for i in indexes]
        centroid=vectors[indexes].mean(axis=0); centroid/=np.linalg.norm(centroid)
        hits=client().query_points('faces',query=centroid.tolist(),limit=1,with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(key='person_id',range=models.Range(gt=0))])).points
        if hits and hits[0].score >= settings.centroid_match_threshold:
            pid=hits[0].payload['person_id']; matched+=1
        else:
            pid=people.create_person(); new+=1
        people.assign_faces(cluster,pid,"cluster")
    return {"pending":len(ids),"new_people":new,"matched_existing":matched}
