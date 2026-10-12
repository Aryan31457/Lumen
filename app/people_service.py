import json
from qdrant_client import models
from .db import conn
from .qdrant_store import client
from .tokens import sanitize_name


def create_person():
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO people DEFAULT VALUES RETURNING person_id"); return cur.fetchone()["person_id"]


def _sync_image(cur, image_id):
    cur.execute("SELECT DISTINCT person_id FROM faces WHERE image_id=%s AND person_id IS NOT NULL", (image_id,))
    ids = sorted(row["person_id"] for row in cur.fetchall())
    cur.execute("DELETE FROM image_people WHERE image_id=%s", (image_id,))
    for pid in ids: cur.execute("INSERT INTO image_people VALUES (%s,%s)", (image_id, pid))
    return ids


def _push_payloads(items):
    for image_id, ids in items.items():
        try: client().set_payload("photos", {"person_ids": ids}, points=[image_id])
        except Exception: pass


def _replace(cur, image_id, old, new):
    cur.execute("UPDATE images SET caption_template=replace(coalesce(caption_template,''),%s,%s), presence_template=replace(coalesce(presence_template,''),%s,%s) WHERE image_id=%s", (old,new,old,new,image_id))


def rename(person_id, new_name):
    with conn() as c, c.cursor() as cur: cur.execute("UPDATE people SET display_name=%s WHERE person_id=%s AND NOT deleted", (sanitize_name(new_name) or None, person_id))


def assign_faces(face_ids, person_id, by="manual"):
    if not face_ids: return
    affected = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id,image_id FROM faces WHERE face_id=ANY(%s)", (face_ids,)); rows = cur.fetchall()
        cur.execute("UPDATE faces SET person_id=%s,assigned_by=%s WHERE face_id=ANY(%s)", (person_id,by,face_ids))
        for row in rows: _replace(cur,row['image_id'],f"{{{{f:{row['face_id']}}}}}",f"{{{{p:{person_id}}}}}")
        for image_id in {row['image_id'] for row in rows}: affected[image_id] = _sync_image(cur,image_id)
    client().set_payload("faces", {"person_id":person_id}, points=face_ids); _push_payloads(affected)


def unassign_faces(face_ids):
    """Clear identity assignments while preserving face placeholders in captions."""
    if not face_ids:
        return
    affected = {}
    rows = []
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id,image_id,person_id FROM faces WHERE face_id=ANY(%s) AND person_id IS NOT NULL", (face_ids,))
        rows = cur.fetchall()
        if not rows:
            return
        cur.execute("UPDATE faces SET person_id=NULL,assigned_by=NULL,assign_sim=NULL WHERE face_id=ANY(%s)",
                    ([row["face_id"] for row in rows],))
        for image_id in {row["image_id"] for row in rows}:
            image_rows = [row for row in rows if row["image_id"] == image_id]
            for row in image_rows:
                cur.execute("SELECT 1 FROM faces WHERE image_id=%s AND person_id=%s LIMIT 1",
                            (image_id, row["person_id"]))
                if cur.fetchone() is None:
                    _replace(cur, image_id, f"{{{{p:{row['person_id']}}}}}", f"{{{{f:{row['face_id']}}}}}")
            affected[image_id] = _sync_image(cur, image_id)
    ids = [row["face_id"] for row in rows]
    client().set_payload("faces", {"person_id": 0}, points=ids)
    _push_payloads(affected)


def merge(keep, drop):
    if keep == drop: return
    affected = {}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT person_id FROM people WHERE person_id IN (%s,%s) AND merged_into IS NULL AND NOT deleted",(keep,drop))
        if len(cur.fetchall()) != 2: raise ValueError("Both people must be active")
        cur.execute("SELECT DISTINCT image_id FROM faces WHERE person_id=%s",(drop,)); images=[r['image_id'] for r in cur.fetchall()]
        cur.execute("UPDATE faces SET person_id=%s,assigned_by='manual' WHERE person_id=%s",(keep,drop)); cur.execute("UPDATE people SET merged_into=%s WHERE person_id=%s",(keep,drop))
        for image_id in images: _replace(cur,image_id,f"{{{{p:{drop}}}}}",f"{{{{p:{keep}}}}}"); affected[image_id]=_sync_image(cur,image_id)
    client().set_payload("faces", {"person_id":keep}, points=models.Filter(must=[models.FieldCondition(key="person_id",match=models.MatchValue(value=drop))])); _push_payloads(affected)


def split(person_id, face_ids):
    new_id=create_person(); affected={}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id,image_id FROM faces WHERE face_id=ANY(%s) AND person_id=%s",(face_ids,person_id)); rows=cur.fetchall()
        cur.execute("UPDATE faces SET person_id=%s,assigned_by='manual' WHERE face_id=ANY(%s)",(new_id,[r['face_id'] for r in rows]))
        for row in rows: affected[row['image_id']]=_sync_image(cur,row['image_id'])
    if rows: client().set_payload("faces", {"person_id":new_id}, points=[r['face_id'] for r in rows])
    _push_payloads(affected); return new_id


def delete_person(person_id):
    affected={}
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id,image_id FROM faces WHERE person_id=%s",(person_id,)); rows=cur.fetchall(); ids=[r['face_id'] for r in rows]
        for row in rows: _replace(cur,row['image_id'],f"{{{{p:{person_id}}}}}","a person")
        cur.execute("DELETE FROM faces WHERE person_id=%s",(person_id,)); cur.execute("UPDATE people SET deleted=TRUE,display_name=NULL WHERE person_id=%s",(person_id,))
        for image_id in {r['image_id'] for r in rows}: affected[image_id]=_sync_image(cur,image_id)
    if ids: client().delete("faces", points_selector=models.PointIdsList(points=ids))
    _push_payloads(affected)
