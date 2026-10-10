import re
from datetime import date
from difflib import get_close_matches
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from qdrant_client import models
from ..config import settings
from ..db import conn
from ..models import get_llm, embed_query, clip_text_vec
from ..qdrant_store import client
from ..tokens import load_people, render, display, summarise_group
from .. import prompts


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


def parse(state):
    plan=get_llm().with_structured_output(SearchPlan).invoke([SystemMessage(content=prompts.QUERY_PARSE_SYSTEM.replace('<<TODAY>>',date.today().isoformat())),HumanMessage(content=state['question'])])
    return {'plan':plan.model_dump()}


def resolve(state):
    ids=[]; unknown=[]
    for name in state['plan']['person_names']:
        matches=find_people(name)
        if not matches: unknown.append(name)
        elif len(matches)==1: ids.append(matches[0]['person_id'])
        else: ids.append(int(interrupt({'type':'disambiguate','name':name,'options':[{'person_id':r['person_id'],'label':r['display_name'] or f"User {r['person_id']}"} for r in matches]})))
    return {'person_ids':ids,'unknown':unknown}


def retrieve(state):
    if state.get('unknown'): return {'candidates':[]}
    plan=state['plan']; must=[]; ids=state.get('person_ids',[])
    if ids: must += [models.FieldCondition(key='person_ids',match=models.MatchValue(value=pid)) for pid in ids] if plan['person_match']=='all' else [models.FieldCondition(key='person_ids',match=models.MatchAny(any=ids))]
    query=plan['semantic_query'].strip(); filter=models.Filter(must=must) if must else None; limit=max(30,plan['limit']*3)
    if not query:
        points,_=client().scroll('photos',scroll_filter=filter,limit=limit,with_payload=True)
        return {'candidates':[{'image_id':p.id,'score':0,'payload':p.payload} for p in points]}
    points=client().query_points('photos',prefetch=[models.Prefetch(query=embed_query(query),using='activity',filter=filter,limit=limit),models.Prefetch(query=clip_text_vec(query),using='clip',filter=filter,limit=limit)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
    return {'candidates':[{'image_id':p.id,'score':p.score,'payload':p.payload} for p in points]}


def respond(state):
    if state.get('unknown'):
        return {'answer':{'message':"I don't know anyone named " + ', '.join(state['unknown']) + ' yet. Name a face group first.','results':[],'low_confidence':False}}
    candidates=state.get('candidates',[])[:state['plan']['limit']]; ids=[x['image_id'] for x in candidates]; people=load_people()
    with conn() as c, c.cursor() as cur:
        cur.execute('SELECT image_id,path,caption_template,taken_at FROM images WHERE image_id=ANY(%s)',(ids,)); rows={r['image_id']:r for r in cur.fetchall()}
    results=[]
    for item in candidates:
        row=rows.get(item['image_id'])
        if row: results.append({'image_id':row['image_id'],'path':row['path'],'caption':render(row['caption_template'],people),'people':summarise_group([display(pid,people) for pid in item['payload'].get('person_ids',[])]),'taken_at':row['taken_at'].isoformat() if row['taken_at'] else None,'confidence':'high'})
    return {'answer':{'message':f"Found {len(results)} photo(s)." if results else "I couldn't find matching photos.",'results':results,'low_confidence':False}}


def build_query_graph(checkpointer=None):
    graph=StateGraph(dict)
    for name,fn in [('parse',parse),('resolve',resolve),('retrieve',retrieve),('respond',respond)]: graph.add_node(name,fn)
    graph.add_edge(START,'parse'); graph.add_edge('parse','resolve'); graph.add_edge('resolve','retrieve'); graph.add_edge('retrieve','respond'); graph.add_edge('respond',END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
