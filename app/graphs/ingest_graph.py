import hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from qdrant_client import models
from . import __init__
from ..config import settings
from ..db import conn
from ..faces import detect, try_assign
from ..captioner import caption_image
from ..models import clip_image_vec, embed_doc
from ..people_service import assign_faces
from ..qdrant_store import client
from ..tokens import presence_template


def _sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda: source.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def ingest_path(path):
    source=Path(path); digest=_sha(path); Path(settings.image_dir).mkdir(parents=True,exist_ok=True)
    with conn() as c, c.cursor() as cur:
        cur.execute('SELECT image_id FROM images WHERE sha256=%s',(digest,)); existing=cur.fetchone()
        if existing: return {'image_id':existing['image_id'],'skipped':True}
        destination=Path(settings.image_dir)/f'{digest}{source.suffix.lower()}'; shutil.copy2(source,destination)
        with Image.open(destination) as pil: width,height=pil.size
        cur.execute("INSERT INTO images(path,sha256,width,height,taken_at,status) VALUES(%s,%s,%s,%s,%s,'new') RETURNING image_id",(str(destination),digest,width,height,datetime.fromtimestamp(destination.stat().st_mtime,timezone.utc)))
        image_id=cur.fetchone()['image_id']
    try:
        image, found=detect(str(destination)); tokens=[]; face_ids=[]
        with conn() as c, c.cursor() as cur:
            for item in found:
                pid,sim=(try_assign(item['embedding']) if item['quality_ok'] else (None,0.0))
                cur.execute("INSERT INTO faces(image_id,person_id,bbox,det_score,blur_var,quality_ok,assign_sim,assigned_by) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING face_id",(image_id,pid,json.dumps(item['bbox']),item['det_score'],item['blur_var'],item['quality_ok'],sim,'auto' if pid else None)); fid=cur.fetchone()['face_id']; face_ids.append((fid,item,pid))
                tokens.append(f'{{{{p:{pid}}}}}' if pid else f'{{{{f:{fid}}}}}')
        for fid,item,pid in face_ids: client().upsert('faces',[models.PointStruct(id=fid,vector=item['embedding'],payload={'person_id':pid or 0,'image_id':image_id})])
        labels=[]; label_tokens={}
        for face_id, item, person_id in face_ids:
            if item['quality_ok']:
                label=f'P{len(labels)+1}'; labels.append(label)
                label_tokens[label]=f'{{{{p:{person_id}}}}}' if person_id else f'{{{{f:{face_id}}}}}'
        cap=caption_image(image,labels)
        template=cap.caption
        for label,token in label_tokens.items(): template=template.replace(label,token)
        p_tokens=[token for token in tokens if token.startswith('{{p:')]
        with conn() as c, c.cursor() as cur:
            cur.execute("UPDATE images SET caption_template=%s,presence_template=%s,activity_text=%s,scene=%s,tags=%s,status='indexed' WHERE image_id=%s",(template,presence_template(p_tokens),cap.activity,cap.scene,json.dumps(cap.tags),image_id))
            cur.execute('SELECT DISTINCT person_id FROM faces WHERE image_id=%s AND person_id IS NOT NULL',(image_id,)); pids=[r['person_id'] for r in cur.fetchall()]
            for pid in pids: cur.execute('INSERT INTO image_people VALUES(%s,%s) ON CONFLICT DO NOTHING',(image_id,pid))
        with Image.open(destination) as pil: clip=clip_image_vec(pil.convert('RGB'))
        client().upsert('photos',[models.PointStruct(id=image_id,vector={'activity':embed_doc((cap.activity+' '+cap.scene).strip()),'clip':clip},payload={'image_id':image_id,'person_ids':pids,'tags':cap.tags,'activity':cap.activity,'taken_at_ts':destination.stat().st_mtime})])
        return {'image_id':image_id,'indexed':True}
    except Exception as exc:
        with conn() as c, c.cursor() as cur: cur.execute("UPDATE images SET status='failed',error=%s WHERE image_id=%s",(str(exc),image_id))
        raise


def ingest_folder(folder):
    extensions={'.jpg','.jpeg','.png','.webp'}; results=[]
    for path in Path(folder).iterdir():
        if path.is_file() and path.suffix.lower() in extensions: results.append(ingest_path(str(path)))
    return {'results':results,'indexed':sum(bool(r.get('indexed')) for r in results)}


def build_ingest_graph():
    """Compatibility entry point; ingestion is intentionally transactional per file."""
    class Graph:
        def invoke(self,state,*_args,**_kwargs): return ingest_path(state['path'])
    return Graph()
