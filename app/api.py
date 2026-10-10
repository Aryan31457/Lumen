import os, shutil, tempfile, uuid
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from langgraph.types import Command
from .db import conn
from .qdrant_store import ensure_collections
from .graphs.ingest_graph import build_ingest_graph
from .graphs.query_graph import build_query_graph
from .clustering import run_cluster_job
from . import people_service as people
from .tokens import load_people, display

app=FastAPI(title='Photo Memory'); ingest_graph=build_ingest_graph(); query_graph=build_query_graph()
class SearchIn(BaseModel): question:str; thread_id:str|None=None
class ResumeIn(BaseModel): thread_id:str; person_id:int
class RenameIn(BaseModel): name:str|None=None
class MergeIn(BaseModel): keep:int; drop:int
class SplitIn(BaseModel): face_ids:list[int]
@app.on_event('startup')
def startup(): ensure_collections()
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
        cur.execute("SELECT p.person_id,count(DISTINCT f.image_id) photos,min(f.face_id) cover_face_id FROM people p JOIN faces f ON f.person_id=p.person_id WHERE p.merged_into IS NULL AND NOT p.deleted GROUP BY p.person_id ORDER BY photos DESC"); rows=cur.fetchall()
    return [{**row,'label':display(row['person_id'],named)} for row in rows]
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
@app.get('/faces/{face_id}/crop')
def crop(face_id:int):
    with conn() as c, c.cursor() as cur: cur.execute('SELECT i.path,f.bbox FROM faces f JOIN images i USING(image_id) WHERE face_id=%s',(face_id,)); row=cur.fetchone()
    if not row: raise HTTPException(404)
    image=cv2.imread(row['path']); x1,y1,x2,y2=map(int,row['bbox']); ok,encoded=cv2.imencode('.jpg',image[y1:y2,x1:x2]); return Response(encoded.tobytes(),media_type='image/jpeg') if ok else Response(status_code=422)
