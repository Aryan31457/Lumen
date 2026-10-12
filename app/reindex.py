import json
from pathlib import Path

import cv2
from qdrant_client import models

from .captioner import caption_image
from .db import conn
from .models import embed_doc
from .qdrant_store import client, ensure_collections
from .search_text import build_search_text
from .tokens import presence_template
from .feedback import ensure_feedback_schema


def reindex_captions(image_ids=None):
    """Regenerate local captions and text vectors for already indexed photos."""
    ensure_collections()
    ensure_feedback_schema()
    with conn() as db, db.cursor() as cur:
        if image_ids:
            cur.execute("SELECT image_id,path FROM images WHERE status='indexed' AND NOT caption_locked AND image_id=ANY(%s) ORDER BY image_id", (image_ids,))
        else:
            cur.execute("SELECT image_id,path FROM images WHERE status='indexed' AND NOT caption_locked ORDER BY image_id")
        images = cur.fetchall()

    updated = skipped = 0
    for image_row in images:
        image_id = image_row["image_id"]
        image = cv2.imread(str(Path(image_row["path"])))
        if image is None:
            skipped += 1
            continue
        with conn() as db, db.cursor() as cur:
            cur.execute(
                "SELECT face_id,person_id,quality_ok FROM faces WHERE image_id=%s ORDER BY face_id",
                (image_id,),
            )
            faces = cur.fetchall()

        labels = []
        label_tokens = {}
        for face in faces:
            if not face["quality_ok"]:
                continue
            label = f"P{len(labels) + 1}"
            labels.append(label)
            pid = face["person_id"]
            label_tokens[label] = f"{{{{p:{pid}}}}}" if pid else f"{{{{f:{face['face_id']}}}}}"

        caption = caption_image(image, labels)
        template = caption.caption
        for label, token in label_tokens.items():
            template = template.replace(label, token)
        search_text = build_search_text(template, caption.activity, caption.scene, caption.tags)
        with conn() as db, db.cursor() as cur:
            cur.execute(
                "UPDATE images SET caption_template=%s,presence_template=%s,activity_text=%s,scene=%s,tags=%s WHERE image_id=%s",
                (template, presence_template([v for v in label_tokens.values() if v.startswith("{{p:")]),
                 caption.activity, caption.scene, json.dumps(caption.tags), image_id),
            )
            cur.execute(
                "SELECT DISTINCT person_id FROM faces WHERE image_id=%s AND person_id IS NOT NULL",
                (image_id,),
            )
            person_ids = [row["person_id"] for row in cur.fetchall()]

        store = client()
        store.set_payload("photos", {
            "person_ids": person_ids, "caption": template, "activity": caption.activity,
            "scene": caption.scene, "tags": caption.tags, "search_text": search_text,
        }, points=[image_id])
        store.update_vectors("photos", points=[models.PointVectors(
            id=image_id, vector={"activity": embed_doc(search_text)},
        )])
        updated += 1

    return {"updated": updated, "skipped": skipped}
