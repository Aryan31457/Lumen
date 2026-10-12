import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from qdrant_client import models
from ..config import settings
from ..db import conn
from ..models import embed_query, clip_text_vec
from ..qdrant_store import client
from ..tokens import load_people, render, display, summarise_group
from ..search_text import search_terms
from ..feedback import search_adjustments, editable_caption


class SearchPlan(BaseModel):
    person_names: list[str] = Field(default_factory=list)
    person_match: Literal['all','any']='all'
    activity: str=''
    scene_terms: list[str]=Field(default_factory=list)
    date_from: str|None=None
    date_to: str|None=None
    semantic_query: str=''
    limit: int=Field(default=12,ge=1,le=50)


def find_people(name):
    label=re.match(r'^\s*user\s*(\d+)\s*$',name,re.I)
    with conn() as c, c.cursor() as cur:
        if label: cur.execute('SELECT person_id,display_name FROM people WHERE person_id=%s AND merged_into IS NULL AND NOT deleted',(int(label.group(1)),))
        else: cur.execute('SELECT person_id,display_name FROM people WHERE merged_into IS NULL AND NOT deleted AND lower(display_name)=lower(%s)',(name,))
        return cur.fetchall()


def _fallback_plan(question):
    people=load_people(); names=[]; remaining=question; today=date.today(); date_from=date_to=None
    for pid,person in people.items():
        label=f"User {pid}"
        for candidate in (person.get('display_name'),label):
            if candidate and re.search(rf'(?<!\w){re.escape(candidate)}(?!\w)',question,re.I):
                names.append(candidate); remaining=re.sub(rf'(?<!\w){re.escape(candidate)}(?!\w)',' ',remaining,flags=re.I)
    explicit=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',remaining)
    try:
        if len(explicit)>1: date_from,date_to=explicit[0],explicit[1]
        elif explicit: date_from=date_to=explicit[0]
        elif re.search(r'\byesterday\b',remaining,re.I): date_from=date_to=(today-timedelta(days=1)).isoformat()
        elif re.search(r'\btoday\b',remaining,re.I): date_from=date_to=today.isoformat()
        elif re.search(r'\blast week\b',remaining,re.I): date_from=(today-timedelta(days=6)).isoformat(); date_to=today.isoformat()
        elif re.search(r'\bthis year\b',remaining,re.I): date_from=date(today.year,1,1).isoformat(); date_to=today.isoformat()
        elif re.search(r'\blast year\b',remaining,re.I): date_from=date(today.year-1,1,1).isoformat(); date_to=date(today.year-1,12,31).isoformat()
    except ValueError:
        date_from=date_to=None
    remaining=re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b(?:yesterday|today|last week|this year|last year)\b',' ',remaining,flags=re.I)
    remaining=re.sub(r'\b(?:show me|find|search for|look for|please|photos?|pictures?|images?|from|on|at|taken|during|of|the|with|can you|could you|i want|give me|near|around)\b',' ',remaining,flags=re.I)
    remaining=' '.join(remaining.split())
    semantic=remaining or (question if not names and not (date_from or date_to) else '')
    person_match='any' if re.search(r'\b(?:either|any of|anyone)\b',question,re.I) else 'all'
    return SearchPlan(person_names=names,person_match=person_match,date_from=date_from,date_to=date_to,semantic_query=semantic).model_dump()


def parse(state):
    # Keep query understanding deterministic: the tiny general text model was
    # frequently emitting malformed plans and slowing every first search.
    return {'plan':_fallback_plan(state['question'])}


def resolve(state):
    ids=[]; unknown=[]
    for name in state['plan']['person_names']:
        matches=find_people(name)
        if not matches: unknown.append(name)
        elif len(matches)==1: ids.append(matches[0]['person_id'])
        else: ids.append(int(interrupt({'type':'disambiguate','name':name,'options':[{'person_id':r['person_id'],'label':r['display_name'] or f"User {r['person_id']}"} for r in matches]})))
    return {'plan':state['plan'],'person_ids':ids,'unknown':unknown}


def retrieve(state):
    if state.get('unknown'): return {'plan':state['plan'],'person_ids':state.get('person_ids',[]),'unknown':state['unknown'],'candidates':[]}
    plan=state['plan']; must=[]; ids=state.get('person_ids',[])
    if ids: must += [models.FieldCondition(key='person_ids',match=models.MatchValue(value=pid)) for pid in ids] if plan['person_match']=='all' else [models.FieldCondition(key='person_ids',match=models.MatchAny(any=ids))]
    date_range={}
    try:
        if plan.get('date_from'): date_range['gte']=datetime.combine(date.fromisoformat(plan['date_from']),time.min,tzinfo=timezone.utc).timestamp()
        if plan.get('date_to'): date_range['lte']=datetime.combine(date.fromisoformat(plan['date_to']),time.max,tzinfo=timezone.utc).timestamp()
    except ValueError:
        date_range={}
    if date_range: must.append(models.FieldCondition(key='taken_at_ts',range=models.Range(**date_range)))
    query=(plan['semantic_query'] or ' '.join([plan.get('activity',''),*plan.get('scene_terms',[])])).strip()
    filter=models.Filter(must=must) if must else None
    limit=min(500,max(40,plan['limit']*settings.search_candidate_multiplier))
    if not query:
        points,_=client().scroll('photos',scroll_filter=filter,limit=limit,with_payload=True)
        feedback=search_adjustments(state.get('question',''))
        candidates=[{'image_id':p.id,'score':feedback.get(p.id,0.0),'payload':p.payload} for p in points]
        candidates.sort(key=lambda item:item['score'],reverse=True)
        return {'plan':plan,'person_ids':ids,'unknown':state.get('unknown',[]),'candidates':candidates}
    qdrant=client()
    text_points=qdrant.query_points('photos',query=embed_query(query),using='activity',query_filter=filter,limit=limit,with_payload=True).points
    clip_points=qdrant.query_points('photos',query=clip_text_vec(query),using='clip',query_filter=filter,limit=limit,with_payload=True).points
    clip_weight=max(0.0,settings.clip_search_weight); text_weight=max(0.0,settings.text_search_weight)
    lexical_weight=max(0.0,settings.lexical_search_weight); rank_constant=60
    denominator=(clip_weight+text_weight)/(rank_constant+1) or 1.0
    scored={}
    payloads={}
    raw_scores={}
    for source,weight,points in (("clip",clip_weight,clip_points),("text",text_weight,text_points)):
        if weight <= 0:
            continue
        for rank,point in enumerate(points,1):
            scored[point.id]=scored.get(point.id,0.0)+weight/(rank_constant+rank)
            payloads[point.id]=point.payload or {}
            raw_scores.setdefault(point.id,{})[source] = point.score
    terms=search_terms(query)
    for image_id,payload in payloads.items():
        metadata=' '.join(str(payload.get(key,'')) for key in ('search_text','caption','activity','scene','tags'))
        matched=len(terms & search_terms(metadata))/len(terms) if terms else 0.0
        scored[image_id]=scored[image_id]/denominator+lexical_weight*matched
    for image_id, adjustment in search_adjustments(state.get('question', query)).items():
        if image_id in scored:
            scored[image_id] += adjustment
    ordered=sorted(scored,key=scored.get,reverse=True)
    candidates=[{'image_id':image_id,'score':scored[image_id],
                 'clip_score':raw_scores.get(image_id,{}).get('clip',0.0),
                 'text_score':raw_scores.get(image_id,{}).get('text',0.0),
                 'payload':payloads[image_id]} for image_id in ordered]
    return {'plan':plan,'person_ids':ids,'unknown':state.get('unknown',[]),'candidates':candidates}


def respond(state):
    if state.get('unknown'):
        return {'answer':{'message':"I don't know anyone named " + ', '.join(state['unknown']) + ' yet. Name a face group first.','results':[],'low_confidence':False,'query':state.get('question','')}}
    candidates=state.get('candidates',[])[:state['plan']['limit']]; ids=[x['image_id'] for x in candidates]; people=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute('SELECT image_id,path,caption_template,taken_at FROM images WHERE image_id=ANY(%s)',(ids,)); rows={r['image_id']:r for r in cur.fetchall()}
    results=[]
    exact_filter=bool(state.get('person_ids') or state['plan'].get('date_from') or state['plan'].get('date_to'))
    for item in candidates:
        row=rows.get(item['image_id'])
        if row:
            clip_score=item.get('clip_score',0.0); text_score=item.get('text_score',0.0)
            if not state['plan'].get('semantic_query'):
                confidence='high' if exact_filter else 'medium'
            elif clip_score >= settings.search_min_clip_confidence and text_score >= settings.search_min_text_confidence:
                confidence='high'
            elif clip_score >= settings.search_min_clip_confidence*0.65 or text_score >= settings.search_min_text_confidence*0.8:
                confidence='medium'
            else:
                confidence='low'
            results.append({'image_id':row['image_id'],'path':row['path'],'caption':render(row['caption_template'],people),'editable_caption':editable_caption(row['caption_template'],row['image_id']),'people':summarise_group([display(pid,people) for pid in item['payload'].get('person_ids',[])]),'taken_at':row['taken_at'].isoformat() if row['taken_at'] else None,'confidence':confidence})
    low_confidence=not results or results[0]['confidence']=='low'
    return {'answer':{'message':f"Found {len(results)} photo(s)." if results else "I couldn't find matching photos.",'results':results,'low_confidence':low_confidence,'query':state.get('question','')}}


def build_query_graph(checkpointer=None):
    graph=StateGraph(dict)
    for name,fn in [('parse',parse),('resolve',resolve),('retrieve',retrieve),('respond',respond)]: graph.add_node(name,fn)
    graph.add_edge(START,'parse'); graph.add_edge('parse','resolve'); graph.add_edge('resolve','retrieve'); graph.add_edge('retrieve','respond'); graph.add_edge('respond',END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
