import requests
import streamlit as st

API='http://localhost:8000'
st.set_page_config(page_title='Photo Memory',layout='wide')
st.title('Photo Memory')
upload, people, search = st.tabs(['Upload','People','Search'])
with upload:
    files=st.file_uploader('Choose photos',type=['jpg','jpeg','png','webp'],accept_multiple_files=True)
    if st.button('Ingest selected photos') and files:
        for file in files: requests.post(f'{API}/ingest',files={'file':(file.name,file.getvalue(),file.type)}).raise_for_status()
        st.success('Photos ingested.')
    if st.button('Cluster unassigned faces'): st.json(requests.post(f'{API}/cluster/run').json())
with people:
    try:
        for person in requests.get(f'{API}/people').json():
            cols=st.columns([1,3,2]); cols[0].image(f"{API}/faces/{person['cover_face_id']}/crop"); cols[1].write(f"{person['label']} — {person['photos']} photos")
            name=cols[2].text_input('Name',value='' if person['label'].startswith('User ') else person['label'],key=f"name{person['person_id']}")
            if cols[2].button('Save',key=f"save{person['person_id']}"): requests.patch(f"{API}/people/{person['person_id']}",json={'name':name}); st.rerun()
    except requests.RequestException: st.info('Start the API to manage people.')
with search:
    question=st.chat_input("Ask about your photos, e.g. 'Aryan swimming'")
    if question: st.session_state.last=requests.post(f'{API}/search',json={'question':question}).json()
    result=st.session_state.get('last')
    if result and 'needs_choice' in result:
        choice=st.radio("Which person?",result['needs_choice']['options'],format_func=lambda x:x['label'])
        if st.button('Confirm'): st.session_state.last=requests.post(f'{API}/search/resume',json={'thread_id':result['thread_id'],'person_id':choice['person_id']}).json(); st.rerun()
    elif result:
        st.write(result['answer']['message']); columns=st.columns(3)
        for index,item in enumerate(result['answer']['results']):
            with columns[index%3]: st.image(f"{API}/images/{item['image_id']}/file"); st.text(item['caption']); st.caption(item['people'])
