import os, re, shutil, tempfile, uuid
from io import BytesIO
from pathlib import Path
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel
from langgraph.types import Command
from .db import conn
from .qdrant_store import ensure_collections
from .graphs.ingest_graph import build_ingest_graph
from .graphs.query_graph import build_query_graph
from .clustering import run_cluster_job
from . import people_service as people
from .tokens import load_people, display, render
from .feedback import ensure_feedback_schema, editable_caption, correct_caption, correct_classification, record_search_feedback

app=FastAPI(title='Photo Memory'); ingest_graph=build_ingest_graph(); query_graph=build_query_graph()
class SearchIn(BaseModel): question:str; thread_id:str|None=None
class ResumeIn(BaseModel): thread_id:str; person_id:int
class RenameIn(BaseModel): name:str|None=None
class MergeIn(BaseModel): keep:int; drop:int
class SplitIn(BaseModel): face_ids:list[int]
class CaptionFeedbackIn(BaseModel): image_id:int; corrected_caption:str
class SearchFeedbackIn(BaseModel): image_id:int; query:str; relevant:bool; rank_position:int|None=None
class ClassificationFeedbackIn(BaseModel): face_id:int; correct_person_id:int|None=None; new_person_name:str|None=None; move_cluster_faces:bool=True
class ProfilePhotoIn(BaseModel): face_id:int
class ExportIn(BaseModel): destination_path:str; confirm:bool=False
@app.on_event('startup')
def startup(): ensure_collections(); ensure_feedback_schema()
@app.post('/ingest')
async def ingest(file:UploadFile=File(...)):
    suffix=os.path.splitext(file.filename or '')[1]
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as out: shutil.copyfileobj(file.file,out); path=out.name
    try: return ingest_graph.invoke({'path':path})
    finally:
        if os.path.exists(path): os.unlink(path)
@app.post('/cluster/run')
def cluster(): return run_cluster_job()
@app.get('/people')
def get_people():
    named=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT p.person_id,count(DISTINCT f.image_id) photos,COALESCE((SELECT pf.face_id FROM faces pf WHERE pf.face_id=p.profile_face_id AND pf.person_id=p.person_id),min(f.face_id)) cover_face_id FROM people p JOIN faces f ON f.person_id=p.person_id WHERE p.merged_into IS NULL AND NOT p.deleted GROUP BY p.person_id,p.profile_face_id ORDER BY photos DESC"); rows=cur.fetchall()
    return [{**row,'label':display(row['person_id'],named)} for row in rows]
@app.get('/people/{pid}/images')
def get_person_images(pid:int):
    named=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT i.image_id,i.caption_template,i.activity_text,i.scene,ARRAY(SELECT f.face_id FROM faces f WHERE f.image_id=i.image_id AND f.person_id=%s ORDER BY f.face_id) person_face_ids FROM images i WHERE i.status='indexed' AND EXISTS (SELECT 1 FROM faces f WHERE f.image_id=i.image_id AND f.person_id=%s) ORDER BY i.taken_at DESC NULLS LAST,i.image_id DESC",(pid,pid))
        rows=cur.fetchall()
    return [{**row,'caption':render(row['caption_template'],named) or row['activity_text'] or row['scene'] or 'Photo','editable_caption':editable_caption(row['caption_template'],row['image_id'])} for row in rows]
@app.post('/people/{pid}/export')
def export_person_images(pid:int,body:ExportIn):
    if not body.confirm: raise HTTPException(400,'Confirm the destination folder before exporting photos.')
    raw_path=os.path.expandvars(os.path.expanduser(body.destination_path.strip()))
    destination=Path(raw_path)
    if not raw_path or not destination.is_absolute(): raise HTTPException(400,'Enter an absolute folder path on this computer.')
    try: destination=destination.resolve()
    except OSError as exc: raise HTTPException(400,'The destination path is invalid.') from exc
    if destination == Path(destination.anchor): raise HTTPException(400,'Choose a folder below the filesystem root.')
    if destination.exists() and not destination.is_dir(): raise HTTPException(400,'The destination exists and is not a folder.')
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT display_name FROM people WHERE person_id=%s AND merged_into IS NULL AND NOT deleted",(pid,))
        person=cur.fetchone()
        if not person: raise HTTPException(404,'Active username not found.')
        cur.execute("SELECT DISTINCT i.image_id,i.path FROM images i JOIN faces f USING(image_id) WHERE f.person_id=%s AND i.status='indexed' ORDER BY i.image_id",(pid,))
        photos=cur.fetchall()
    missing=[row['image_id'] for row in photos if not Path(row['path']).is_file()]
    if missing: raise HTTPException(409,f"Some source photo files are missing: {missing}")
    destination.mkdir(parents=True,exist_ok=True)
    label=person['display_name'] or f'User {pid}'
    slug=re.sub(r'[^A-Za-z0-9._-]+','_',label).strip('._-') or f'User_{pid}'
    copied=[]
    try:
        for row in photos:
            source=Path(row['path'])
            filename=f'{slug}_{row["image_id"]:06d}{source.suffix.lower() or ".jpg"}'
            target=destination/filename
            if source.resolve()==target.resolve(): continue
            shutil.copy2(source,target)
            copied.append(filename)
    except OSError as exc: raise HTTPException(500,f'Could not copy photos to the selected folder: {exc}') from exc
    return {'person_id':pid,'folder':str(destination),'copied':len(copied),'files':copied}
@app.get('/images')
def get_images(limit:int=30,offset:int=0):
    """Return a page of every successfully indexed image, including unassigned faces."""
    if limit < 1 or limit > 100: raise HTTPException(400,'limit must be between 1 and 100.')
    if offset < 0: raise HTTPException(400,'offset cannot be negative.')
    named=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT count(*) total FROM images WHERE status='indexed'")
        total=cur.fetchone()['total']
        cur.execute("""SELECT i.image_id,i.caption_template,i.activity_text,i.scene,i.taken_at,
            ARRAY(SELECT DISTINCT f.person_id FROM faces f WHERE f.image_id=i.image_id AND f.person_id IS NOT NULL ORDER BY f.person_id) person_ids
            FROM images i WHERE i.status='indexed'
            ORDER BY i.taken_at DESC NULLS LAST,i.image_id DESC LIMIT %s OFFSET %s""",(limit,offset))
        rows=cur.fetchall()
    items=[]
    for row in rows:
        person_ids=row['person_ids'] or []
        items.append({**row,'people':[display(pid,named) for pid in person_ids],
            'caption':render(row['caption_template'],named) or row['activity_text'] or row['scene'] or 'Photo',
            'editable_caption':editable_caption(row['caption_template'],row['image_id'])})
    return {'total':total,'limit':limit,'offset':offset,'items':items}
@app.post('/feedback/caption')
def caption_feedback(body:CaptionFeedbackIn):
    try: return correct_caption(body.image_id,body.corrected_caption)
    except LookupError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
@app.post('/feedback/search')
def search_feedback(body:SearchFeedbackIn):
    try: return record_search_feedback(body.image_id,body.query,body.relevant,body.rank_position)
    except LookupError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
@app.post('/feedback/classification')
def classification_feedback(body:ClassificationFeedbackIn):
    try: return correct_classification(body.face_id,body.correct_person_id,body.new_person_name,body.move_cluster_faces)
    except LookupError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
@app.put('/people/{pid}/profile-photo')
def profile_photo(pid:int,body:ProfilePhotoIn):
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM people WHERE person_id=%s AND merged_into IS NULL AND NOT deleted",(pid,))
        if not cur.fetchone(): raise HTTPException(404,'Active username not found.')
        cur.execute("SELECT 1 FROM faces WHERE face_id=%s AND person_id=%s",(body.face_id,pid))
        if not cur.fetchone(): raise HTTPException(400,'Choose a face currently assigned to this username.')
        cur.execute("UPDATE people SET profile_face_id=%s WHERE person_id=%s",(body.face_id,pid))
    return {'person_id':pid,'face_id':body.face_id,'updated':True}
@app.patch('/people/{pid}')
def rename(pid:int,body:RenameIn): people.rename(pid,body.name); return {'ok':True}
@app.post('/people/merge')
def merge(body:MergeIn): people.merge(body.keep,body.drop); return {'ok':True}
@app.post('/people/{pid}/split')
def split(pid:int,body:SplitIn): return {'new_person_id':people.split(pid,body.face_ids)}
@app.delete('/people/{pid}')
def delete(pid:int): people.delete_person(pid); return {'ok':True}
def pack(out,tid): return {'thread_id':tid,'needs_choice':out['__interrupt__'][0].value} if '__interrupt__' in out else {'thread_id':tid,'answer':out['answer']}
@app.post('/search')
def search(body:SearchIn):
    tid=body.thread_id or str(uuid.uuid4()); return pack(query_graph.invoke({'question':body.question},{'configurable':{'thread_id':tid}}),tid)
@app.post('/search/resume')
def resume(body:ResumeIn): return pack(query_graph.invoke(Command(resume=body.person_id),{'configurable':{'thread_id':body.thread_id}}),body.thread_id)
@app.get('/images/{image_id}/file')
def image_file(image_id:int):
    with conn() as c, c.cursor() as cur: cur.execute('SELECT path FROM images WHERE image_id=%s',(image_id,)); row=cur.fetchone()
    if not row: raise HTTPException(404)
    return FileResponse(row['path'])
@app.get('/images/{image_id}/thumbnail')
def image_thumbnail(image_id:int):
    with conn() as c, c.cursor() as cur: cur.execute('SELECT path FROM images WHERE image_id=%s AND status=\'indexed\'',(image_id,)); row=cur.fetchone()
    if not row: raise HTTPException(404)
    try:
        with Image.open(row['path']) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
            image=ImageOps.contain(image,(640,480),method=Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(640,480),(24,24,24))
            canvas.paste(image,((640-image.width)//2,(480-image.height)//2))
            output=BytesIO(); canvas.save(output,format='JPEG',quality=85,optimize=True)
        return Response(output.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=3600'})
    except (OSError,ValueError) as exc: raise HTTPException(422,'Could not make a thumbnail for this photo.') from exc
@app.get('/images/{image_id}/faces')
def image_faces(image_id:int):
    named=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT face_id,person_id,quality_ok FROM faces WHERE image_id=%s ORDER BY face_id",(image_id,)); rows=cur.fetchall()
    return [{**row,'label':display(row['person_id'],named) if row['person_id'] else 'Unidentified face','crop_url':'/faces/%s/crop' % row['face_id']} for row in rows]
@app.get('/faces/{face_id}/crop')
def crop(face_id:int):
    with conn() as c, c.cursor() as cur: cur.execute('SELECT i.path,f.bbox FROM faces f JOIN images i USING(image_id) WHERE face_id=%s',(face_id,)); row=cur.fetchone()
    if not row: raise HTTPException(404)
    image=cv2.imread(row['path']); x1,y1,x2,y2=map(int,row['bbox']); ok,encoded=cv2.imencode('.jpg',image[y1:y2,x1:x2])
    if not ok: return Response(status_code=422)
    with Image.open(BytesIO(encoded.tobytes())) as face:
        face=ImageOps.contain(face.convert('RGB'),(320,320),method=Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(320,320),(24,24,24)); canvas.paste(face,((320-face.width)//2,(320-face.height)//2))
        output=BytesIO(); canvas.save(output,format='JPEG',quality=88)
    return Response(output.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=3600'})
