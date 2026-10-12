"""Persistent human corrections and lightweight online search reranking."""
import re

from qdrant_client import models

from .config import settings
from .db import conn
from .models import embed_doc
from .qdrant_store import client
from .search_text import build_search_text, search_terms
from .tokens import TOKEN, presence_template, sanitize_name

USER_TOKEN = re.compile(r"\{\s*user(?:\s*(\d+))?\s*\}", re.IGNORECASE)


def ensure_feedback_schema():
    """Apply this feature's additive migration for already-created databases."""
    with conn() as db, db.cursor() as cur:
        cur.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS caption_locked BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute("ALTER TABLE people ADD COLUMN IF NOT EXISTS profile_face_id INT REFERENCES faces(face_id) ON DELETE SET NULL")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
              feedback_id BIGSERIAL PRIMARY KEY,
              feedback_type TEXT NOT NULL CONSTRAINT feedback_feedback_type_check CHECK (feedback_type IN ('caption', 'search', 'classification')),
              image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
              face_id INT REFERENCES faces(face_id) ON DELETE CASCADE,
              wrong_person_id INT REFERENCES people(person_id),
              correct_person_id INT REFERENCES people(person_id),
              corrected_caption TEXT,
              search_query TEXT,
              relevant BOOLEAN,
              rank_position INT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              CONSTRAINT feedback_check CHECK ((feedback_type = 'caption' AND corrected_caption IS NOT NULL)
                  OR (feedback_type = 'search' AND search_query IS NOT NULL AND relevant IS NOT NULL)
                  OR (feedback_type = 'classification' AND face_id IS NOT NULL AND correct_person_id IS NOT NULL))
            )
        """)
        cur.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS face_id INT REFERENCES faces(face_id) ON DELETE CASCADE")
        cur.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS wrong_person_id INT REFERENCES people(person_id)")
        cur.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS correct_person_id INT REFERENCES people(person_id)")
        # Upgrade constraints on databases created by the previous feedback version.
        cur.execute("ALTER TABLE feedback DROP CONSTRAINT IF EXISTS feedback_feedback_type_check")
        cur.execute("ALTER TABLE feedback ADD CONSTRAINT feedback_feedback_type_check CHECK (feedback_type IN ('caption','search','classification'))")
        cur.execute("ALTER TABLE feedback DROP CONSTRAINT IF EXISTS feedback_check")
        cur.execute("ALTER TABLE feedback ADD CONSTRAINT feedback_check CHECK ((feedback_type='caption' AND corrected_caption IS NOT NULL) OR (feedback_type='search' AND search_query IS NOT NULL AND relevant IS NOT NULL) OR (feedback_type='classification' AND face_id IS NOT NULL AND correct_person_id IS NOT NULL))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_search_image ON feedback(feedback_type, image_id, created_at DESC)")


def _image_tokens(image_id):
    with conn() as db, db.cursor() as cur:
        cur.execute("SELECT face_id,person_id FROM faces WHERE image_id=%s AND quality_ok ORDER BY face_id", (image_id,))
        faces = cur.fetchall()
    result = []
    seen = set()
    for face in faces:
        token = f"{{{{p:{face['person_id']}}}}}" if face['person_id'] else f"{{{{f:{face['face_id']}}}}}"
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


def editable_caption(template, image_id=None):
    """Show stable, readable {userN} placeholders to caption editors."""
    tokens = _image_tokens(image_id) if image_id is not None else []
    labels = {token: f"{{user{index + 1}}}" for index, token in enumerate(tokens)}
    def replace(match):
        token = match.group(0)
        if token not in labels:
            labels[token] = f"{{user{len(labels) + 1}}}"
        return labels[token]
    return TOKEN.sub(replace, template or "")


def _caption_template(image_id, caption):
    tokens = _image_tokens(image_id)
    # {user} is a convenient alias for the first detected person.
    def replace(match):
        index = int(match.group(1) or 1) - 1
        return tokens[index] if 0 <= index < len(tokens) else match.group(0)
    return USER_TOKEN.sub(replace, caption).strip()


def correct_caption(image_id, caption):
    caption = (caption or "").strip()
    if not caption or len(caption) > 1000:
        raise ValueError("Caption must contain 1 to 1000 characters.")
    template = _caption_template(image_id, caption)
    with conn() as db, db.cursor() as cur:
        cur.execute("SELECT scene,tags,status FROM images WHERE image_id=%s", (image_id,))
        row = cur.fetchone()
        if not row or row['status'] != 'indexed':
            raise LookupError("Indexed photo not found.")
        # The corrected caption replaces the model's possibly-wrong activity
        # description in the searchable document as well as in the UI.
        text = build_search_text(template, template, row['scene'], row['tags'])
        vector = embed_doc(text)
        cur.execute("INSERT INTO feedback(feedback_type,image_id,corrected_caption) VALUES('caption',%s,%s)", (image_id, caption))
        cur.execute("UPDATE images SET caption_template=%s,activity_text=%s,caption_locked=TRUE WHERE image_id=%s", (template, template, image_id))
    store = client()
    store.set_payload("photos", {"caption": template, "activity": template, "search_text": text}, points=[image_id])
    store.update_vectors("photos", points=[models.PointVectors(id=image_id, vector={"activity": vector})])
    return {"image_id": image_id, "caption_template": template, "caption": caption, "locked": True}


def correct_classification(face_id, correct_person_id=None, new_person_name=None,
                          move_cluster_faces=True):
    """Move a corrected face, and optionally its photo's other same-cluster faces."""
    new_name = sanitize_name(new_person_name) if new_person_name else None
    if correct_person_id is None and not new_name:
        raise ValueError("Select an existing username or provide a new one.")
    if correct_person_id is not None and new_name:
        raise ValueError("Choose an existing username or create a new one, not both.")

    with conn() as db, db.cursor() as cur:
        cur.execute("SELECT image_id,person_id FROM faces WHERE face_id=%s", (face_id,))
        face = cur.fetchone()
        if not face:
            raise LookupError("Face not found.")
        image_id, wrong_pid = face['image_id'], face['person_id']
        if new_name:
            cur.execute("INSERT INTO people(display_name) VALUES(%s) RETURNING person_id", (new_name,))
            correct_person_id = cur.fetchone()['person_id']
            target_label = new_name
        else:
            cur.execute("SELECT display_name FROM people WHERE person_id=%s AND merged_into IS NULL AND NOT deleted", (correct_person_id,))
            target = cur.fetchone()
            if not target:
                raise LookupError("Active username not found.")
            target_label = target['display_name'] or f"User {correct_person_id}"
        if wrong_pid == correct_person_id:
            return {"face_id": face_id, "image_id": image_id, "person_id": correct_person_id,
                    "username": target_label, "changed": False}

        cur.execute("SELECT caption_template FROM images WHERE image_id=%s", (image_id,))
        image = cur.fetchone()
        template = image['caption_template'] or ""
        new_token = f"{{{{p:{correct_person_id}}}}}"
        changed_face_ids = [face_id]
        if wrong_pid is None:
            template = template.replace(f"{{{{f:{face_id}}}}}", new_token)
        else:
            old_token = f"{{{{p:{wrong_pid}}}}}"
            cur.execute("SELECT face_id FROM faces WHERE image_id=%s AND person_id=%s ORDER BY face_id", (image_id, wrong_pid))
            old_faces = [row['face_id'] for row in cur.fetchall()]
            if move_cluster_faces:
                changed_face_ids = old_faces or [face_id]
                # The photo should disappear from its old cluster completely.
                template = template.replace(old_token, new_token)
            else:
                ordinal = old_faces.index(face_id) + 1 if face_id in old_faces else 1
                parts = template.split(old_token)
                if len(parts) > 1:
                    occurrence = ordinal if len(parts) - 1 >= ordinal else 1
                    template = old_token.join(parts[:occurrence]) + new_token + old_token.join(parts[occurrence:])

        if face_id not in changed_face_ids:
            changed_face_ids.append(face_id)
        cur.execute("UPDATE faces SET person_id=%s,assigned_by='human_feedback',assign_sim=1.0 WHERE face_id=ANY(%s)", (correct_person_id, changed_face_ids))
        cur.executemany("INSERT INTO feedback(feedback_type,image_id,face_id,wrong_person_id,correct_person_id) VALUES('classification',%s,%s,%s,%s)", [(image_id, changed_id, wrong_pid, correct_person_id) for changed_id in changed_face_ids])
        cur.execute("UPDATE people SET profile_face_id=NULL WHERE profile_face_id=ANY(%s) AND person_id<>%s", (changed_face_ids, correct_person_id))
        cur.execute("SELECT DISTINCT person_id FROM faces WHERE image_id=%s AND person_id IS NOT NULL ORDER BY person_id", (image_id,))
        person_ids = [row['person_id'] for row in cur.fetchall()]
        cur.execute("DELETE FROM image_people WHERE image_id=%s", (image_id,))
        cur.executemany("INSERT INTO image_people(image_id,person_id) VALUES(%s,%s)", [(image_id, pid) for pid in person_ids])
        cur.execute("UPDATE images SET caption_template=%s,presence_template=%s WHERE image_id=%s", (template, presence_template([f"{{{{p:{pid}}}}}" for pid in person_ids]), image_id))

    store = client()
    face_points = store.retrieve("faces", ids=changed_face_ids, with_payload=True)
    forbidden_ids = set()
    for point in face_points:
        forbidden = set((point.payload or {}).get("forbidden_person_ids", []))
        if wrong_pid is not None:
            forbidden.add(wrong_pid)
        forbidden.discard(correct_person_id)
        forbidden_ids.update(forbidden)
        store.set_payload("faces", {
            "person_id": correct_person_id,
            "human_verified": True,
            "forbidden_person_ids": sorted(forbidden),
        }, points=[point.id])
    store.set_payload("photos", {"person_ids": person_ids, "caption": template}, points=[image_id])
    return {"face_id": face_id, "image_id": image_id, "person_id": correct_person_id,
            "username": target_label, "changed": True, "moved_faces": len(changed_face_ids),
            "wrong_person_id": wrong_pid, "removed_from_wrong_cluster": wrong_pid is None or wrong_pid not in person_ids,
            "forbidden_person_ids": sorted(forbidden_ids)}


def record_search_feedback(image_id, query, relevant, rank_position=None):
    query = (query or "").strip()
    if not query or len(query) > 500:
        raise ValueError("Search query must contain 1 to 500 characters.")
    with conn() as db, db.cursor() as cur:
        cur.execute("SELECT 1 FROM images WHERE image_id=%s AND status='indexed'", (image_id,))
        if not cur.fetchone():
            raise LookupError("Indexed photo not found.")
        cur.execute(
            "INSERT INTO feedback(feedback_type,image_id,search_query,relevant,rank_position) VALUES('search',%s,%s,%s,%s)",
            (image_id, query, relevant, rank_position),
        )
    return {"image_id": image_id, "recorded": True, "relevant": relevant}


def search_adjustments(query):
    """Return bounded positive/negative relevance signals for similar queries."""
    query_terms = search_terms(query)
    if not query_terms:
        return {}
    with conn() as db, db.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (image_id, lower(trim(search_query))) image_id,search_query,relevant
            FROM feedback WHERE feedback_type='search'
            ORDER BY image_id,lower(trim(search_query)),created_at DESC,feedback_id DESC
        """)
        rows = cur.fetchall()
    totals = {}
    for row in rows:
        previous_terms = search_terms(row['search_query'])
        if not previous_terms:
            continue
        overlap = len(query_terms & previous_terms) / len(query_terms | previous_terms)
        if overlap < 0.5:
            continue
        signals, similarities = totals.setdefault(row['image_id'], ([], []))
        signals.append(1.0 if row['relevant'] else -1.0)
        similarities.append(overlap)
    weight = max(0.0, settings.search_feedback_weight)
    return {
        image_id: weight * sum(signal * similarity for signal, similarity in zip(signals, similarities)) / len(signals)
        for image_id, (signals, similarities) in totals.items()
    }
