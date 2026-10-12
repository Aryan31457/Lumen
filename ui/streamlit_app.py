from urllib.parse import urlencode
from pathlib import Path
import re

import requests
import streamlit as st

API='http://localhost:8000'
st.set_page_config(page_title='Photo Memory',layout='wide')

def linked_photo(image_id, tab, caption, **query):
    captions=st.session_state.setdefault('photo_captions',{})
    captions[str(image_id)]=caption
    params={'main_tab':tab,'photo':image_id,**query}
    return f"{API}/images/{image_id}/thumbnail",f"?{urlencode(params)}"

def clear_photo_query():
    st.query_params.pop('photo',None)

@st.dialog('Photo',width='large',on_dismiss=clear_photo_query)
def show_photo(image_id,caption):
    st.image(f"{API}/images/{image_id}/file",alt=caption or 'Selected photo',width='stretch')
    if caption: st.caption(caption)
    if st.button('Close photo',key=f'close_photo_{image_id}',icon=':material/close:'):
        st.query_params.pop('photo',None)
        st.rerun()

def classification_controls(image_id, context_key, people_rows):
    state_key=f"classify_open_{context_key}_{image_id}"
    status_key=f"classification_status_{context_key}_{image_id}"
    if st.session_state.get(status_key): st.caption(st.session_state[status_key])
    if st.button('Correct person classification',key=f"classify_toggle_{context_key}_{image_id}"):
        st.session_state[state_key]=not st.session_state.get(state_key,False)
        st.rerun()
    if not st.session_state.get(state_key): return
    try:
        response=requests.get(f'{API}/images/{image_id}/faces',timeout=15)
        response.raise_for_status()
        faces=response.json()
    except requests.RequestException as exc:
        st.error(f"Could not load faces for correction: {exc}")
        return
    if not faces:
        st.info('No faces were detected in this photo.')
        return
    face_by_id={face['face_id']:face for face in faces}
    face_id=st.selectbox('Which face is misclassified?',list(face_by_id),format_func=lambda fid:f"Face {fid}: {face_by_id[fid]['label']}",key=f"classify_face_{context_key}_{image_id}")
    selected_face=face_by_id[face_id]
    st.image(f"{API}{selected_face['crop_url']}",caption=selected_face['label'],alt='Selected face to correct',width=120)
    targets=[person['person_id'] for person in people_rows]+[-1]
    target=st.selectbox('Correct username (all existing usernames)',targets,format_func=lambda pid:next((person['label'] for person in people_rows if person['person_id']==pid),'Create a new username'),key=f"classify_target_{context_key}_{image_id}")
    new_name=st.text_input('New username',key=f"classify_new_name_{context_key}_{image_id}") if target==-1 else None
    move_cluster=st.checkbox('Move all faces in this photo from the old username into the selected username',value=True,key=f"classify_all_faces_{context_key}_{image_id}")
    if move_cluster: st.caption('The photo will be removed from the old username cluster.')
    else: st.caption('Only the selected face moves. Other faces can keep the photo in the old username cluster.')
    if st.button('Save classification correction',key=f"classify_save_{context_key}_{image_id}"):
        payload={'face_id':face_id,'correct_person_id':target if target!=-1 else None,'new_person_name':new_name if target==-1 else None,'move_cluster_faces':move_cluster}
        try:
            result=requests.post(f'{API}/feedback/classification',json=payload,timeout=30)
            result.raise_for_status()
            saved=result.json()
            if saved['changed']:
                moved=f"Moved {saved.get('moved_faces',1)} face(s) into {saved['username']}."
                cluster_status=' The photo was removed from the previous cluster.' if saved['removed_from_wrong_cluster'] else ' Other faces keep the photo linked to its previous cluster.'
                st.session_state[status_key]=moved+cluster_status+' Future matches learn from this correction.'
            else:
                st.session_state[status_key]=f"This face is already classified as {saved['username']}."
            st.session_state[state_key]=False
            st.rerun()
        except requests.RequestException as exc: st.error(f"Could not save classification correction: {exc}")

st.title('Photo Memory')
try:
    people_rows=requests.get(f'{API}/people',timeout=15).json()
except requests.RequestException:
    people_rows=[]
upload, library, people, search = st.tabs(['Upload','All photos','People','Search'],key='main_tab',bind='query-params')
with upload:
    files=st.file_uploader('Choose photos',type=['jpg','jpeg','png','webp'],accept_multiple_files=True)
    if st.button('Ingest selected photos') and files:
        for file in files: requests.post(f'{API}/ingest',files={'file':(file.name,file.getvalue(),file.type)}).raise_for_status()
        st.success('Photos ingested.')
    if st.button('Cluster unassigned faces'): st.json(requests.post(f'{API}/cluster/run').json())
with library:
    page_size=12
    total_pages=max(1,(st.session_state.get('library_total',0)+page_size-1)//page_size)
    page=st.session_state.get('library_page',0)
    try:
        response=requests.get(f'{API}/images',params={'limit':page_size,'offset':page*page_size},timeout=30)
        response.raise_for_status()
        library_data=response.json()
        st.session_state.library_total=library_data['total']
        total_pages=max(1,(library_data['total']+page_size-1)//page_size)
        page=min(page,total_pages-1)
        if page != library_data['offset']//page_size:
            st.session_state.library_page=page
            st.rerun()
        st.caption(f"{library_data['total']} indexed photos · page {page+1} of {total_pages}")
        previous,next_page=st.columns(2)
        if previous.button('← Previous',disabled=page==0,key='library_previous'):
            st.session_state.library_page=page-1; st.rerun()
        if next_page.button('Next →',disabled=page>=total_pages-1,key='library_next'):
            st.session_state.library_page=page+1; st.rerun()
        photos=library_data['items']
        if not photos: st.info('No indexed photos yet. Upload photos in the Upload tab.')
        photo_columns=st.columns(3)
        for index,photo in enumerate(photos):
            with photo_columns[index%3]:
                image_id=photo['image_id']
                image_url,image_link=linked_photo(image_id,'All photos',photo['caption'])
                st.image(image_url,caption=photo['caption'],alt=f"Library photo: {photo['caption']}",link=image_link)
                st.caption('People: '+(', '.join(photo['people']) if photo['people'] else 'No assigned username'))
                feedback_key=f"caption_feedback_{image_id}"
                if st.session_state.get(feedback_key): st.caption(st.session_state[feedback_key])
                with st.form(f"library_caption_form_{image_id}"):
                    corrected=st.text_area('Correct caption; use {user1}, {user2}, etc. for people.',value=photo['editable_caption'],key=f"library_caption_edit_{image_id}_{abs(hash(photo['editable_caption']))}")
                    saved=st.form_submit_button('Save caption correction')
                if saved:
                    try:
                        result=requests.post(f'{API}/feedback/caption',json={'image_id':image_id,'corrected_caption':corrected},timeout=60)
                        result.raise_for_status()
                        st.session_state[feedback_key]='Caption correction saved and will be used in future searches.'
                        st.rerun()
                    except requests.RequestException as exc: st.error(f"Could not save caption correction: {exc}")
                classification_controls(image_id,'library',people_rows)
    except requests.RequestException as exc:
        st.error(f'Could not load the photo library: {exc}')
with people:
    try:
        selected_value=st.query_params.get('person')
        selected_person=next((row for row in people_rows if str(row['person_id'])==selected_value),None)
        if selected_person:
            person_id=selected_person['person_id']
            if st.button('All people',icon=':material/arrow_back:'):
                st.query_params.pop('person',None); st.query_params.pop('photo',None); st.rerun()
            st.header(selected_person['label'],icon=':material/person:')
            st.caption(f"{selected_person['photos']} photos")
            with st.expander('Edit username'):
                name=st.text_input('Name',value='' if selected_person['label'].startswith('User ') else selected_person['label'],key=f"name{person_id}")
                if st.button('Save name',key=f"save{person_id}"):
                    response=requests.patch(f"{API}/people/{person_id}",json={'name':name},timeout=20)
                    response.raise_for_status(); st.rerun()
            photos_response=requests.get(f"{API}/people/{person_id}/images",timeout=30)
            photos_response.raise_for_status()
            photos=photos_response.json()
            if not photos: st.info('No indexed photos are assigned to this person yet.')
            with st.expander('Copy this person’s photos to a folder'):
                safe_label=re.sub(r'[^A-Za-z0-9._-]+','_',selected_person['label']).strip('._-') or f"User_{person_id}"
                default_folder=Path.home()/'Pictures'/'Photo Memory'/safe_label
                export_path=st.text_input('Folder path on this computer',value=str(default_folder),key=f'export_path_{person_id}')
                st.caption('This folder will be created on the computer running Photo Memory.')
                st.code(export_path or 'Enter a folder path')
                confirm_key=f"export_confirm_{person_id}_{abs(hash(export_path))}"
                confirmed=st.checkbox(f'Confirm creating or using this folder and copying {len(photos)} photos',key=confirm_key)
                export_status_key=f'export_status_{person_id}'
                if st.session_state.get(export_status_key): st.success(st.session_state[export_status_key])
                if st.button('Create folder and copy photos',key=f'export_button_{person_id}',disabled=not confirmed or not export_path.strip(),icon=':material/folder_copy:'):
                    try:
                        response=requests.post(f'{API}/people/{person_id}/export',json={'destination_path':export_path,'confirm':True},timeout=120)
                        response.raise_for_status()
                        result=response.json()
                        st.session_state[export_status_key]=f"Copied {result['copied']} photos to {result['folder']}"
                        st.rerun()
                    except requests.RequestException as exc:
                        detail=exc.response.text if exc.response is not None else str(exc)
                        st.error(f'Could not copy photos: {detail}')
            photo_columns=st.columns(3)
            for index,photo in enumerate(photos):
                with photo_columns[index%3]:
                    image_id=photo['image_id']
                    image_url,image_link=linked_photo(image_id,'People',photo['caption'],person=person_id)
                    st.image(image_url,caption=photo['caption'],alt=f"Photo associated with {selected_person['label']}",link=image_link)
                    feedback_key=f"caption_feedback_{image_id}"
                    if st.session_state.get(feedback_key): st.caption(st.session_state[feedback_key])
                    with st.expander('Photo actions'):
                        with st.form(f"caption_form_{person_id}_{image_id}"):
                            corrected=st.text_area('Correct caption; use {user1}, {user2}, etc. for people.',value=photo['editable_caption'],key=f"caption_edit_{person_id}_{image_id}_{abs(hash(photo['editable_caption']))}")
                            saved=st.form_submit_button('Save caption correction')
                        if saved:
                            try:
                                response=requests.post(f'{API}/feedback/caption',json={'image_id':image_id,'corrected_caption':corrected},timeout=60)
                                response.raise_for_status()
                                st.session_state[feedback_key]='Caption correction saved and will be used in future searches.'
                                st.rerun()
                            except requests.RequestException as exc: st.error(f"Could not save caption correction: {exc}")
                        profile_key=f"profile_photo_status_{person_id}_{image_id}"
                        if st.session_state.get(profile_key): st.caption(st.session_state[profile_key])
                        profile_faces=photo.get('person_face_ids') or []
                        if profile_faces and st.button('Use as profile photo',key=f"set_profile_{person_id}_{image_id}"):
                            try:
                                response=requests.put(f"{API}/people/{person_id}/profile-photo",json={'face_id':profile_faces[0]},timeout=30)
                                response.raise_for_status(); st.session_state[profile_key]='Profile photo updated.'; st.rerun()
                            except requests.RequestException as exc: st.error(f"Could not update profile photo: {exc}")
                        classification_controls(image_id,f"people_{person_id}",people_rows)
        else:
            st.subheader('People')
            st.caption('Select a face to see that person’s photos.')
            if not people_rows: st.info('No identified people yet. Ingest photos or cluster unassigned faces.')
            else:
                face_columns=st.columns(4)
                for index,person in enumerate(people_rows):
                    with face_columns[index%4]:
                        with st.container(border=True):
                            st.image(f"{API}/faces/{person['cover_face_id']}/crop",alt=f"Face for {person['label']}",link=f"?{urlencode({'main_tab':'People','person':person['person_id']})}",width=180)
                            st.markdown(f"**{person['label']}**")
                            st.caption(f"{person['photos']} photos")
                            if st.button('View photos',key=f"open_person_{person['person_id']}",icon=':material/photo_library:'):
                                st.query_params['person']=str(person['person_id']); st.rerun()
    except requests.RequestException as exc:
        st.error(f'Could not load people or their photos: {exc}')
with search:
    question=st.chat_input("Ask about your photos, e.g. 'Aryan swimming'")
    if question:
        try:
            response=requests.post(f'{API}/search',json={'question':question},timeout=180)
            response.raise_for_status()
            st.session_state.last=response.json()
        except requests.RequestException as exc:
            st.error(f"Search failed: {exc}")
    result=st.session_state.get('last')
    if result and 'needs_choice' in result:
        choice=st.radio("Which person?",result['needs_choice']['options'],format_func=lambda x:x['label'])
        if st.button('Confirm'): st.session_state.last=requests.post(f'{API}/search/resume',json={'thread_id':result['thread_id'],'person_id':choice['person_id']}).json(); st.rerun()
    elif result:
        answer=result['answer']
        st.write(answer['message'])
        if answer.get('low_confidence'):
            st.warning('These matches may be broad. Add a visible action, object, or setting to narrow the search.')
        columns=st.columns(3)
        for index,item in enumerate(answer['results']):
            with columns[index%3]:
                image_id=item['image_id']
                image_url,image_link=linked_photo(image_id,'Search',item['caption'])
                st.image(image_url,caption=item['caption'],alt=f"Search result: {item['caption']}",link=image_link)
                st.caption(f"{item['people']} · {item['confidence'].title()} match")
                feedback_key=f"search_feedback_{image_id}_{abs(hash(answer.get('query','')))}"
                if st.session_state.get(feedback_key): st.caption(st.session_state[feedback_key])
                left,right=st.columns(2)
                for column,relevant,label in ((left,True,'Relevant'),(right,False,'Not relevant')):
                    if column.button(label,key=f"rate_{image_id}_{relevant}_{abs(hash(answer.get('query','')))}"):
                        try:
                            response=requests.post(f'{API}/feedback/search',json={'image_id':image_id,'query':answer.get('query',''),'relevant':relevant,'rank_position':index+1},timeout=30)
                            response.raise_for_status()
                            st.session_state[feedback_key]='Thanks. This will influence similar searches.'
                            st.rerun()
                        except requests.RequestException as exc: st.error(f"Could not save search feedback: {exc}")
                with st.expander('Correct this caption'):
                    with st.form(f"search_caption_form_{image_id}"):
                        corrected=st.text_area('Caption; use {user1}, {user2}, etc. for people.',value=item['editable_caption'],key=f"search_caption_edit_{image_id}_{abs(hash(item['editable_caption']))}")
                        saved=st.form_submit_button('Save caption correction')
                    if saved:
                        try:
                            response=requests.post(f'{API}/feedback/caption',json={'image_id':image_id,'corrected_caption':corrected},timeout=60)
                            response.raise_for_status()
                            st.session_state[f"caption_feedback_{image_id}"]='Caption correction saved and will be used in future searches.'
                            st.rerun()
                        except requests.RequestException as exc: st.error(f"Could not save caption correction: {exc}")
                classification_controls(image_id,'search',people_rows)

photo_value=st.query_params.get('photo')
if photo_value and photo_value.isdigit():
    photo_caption=st.session_state.get('photo_captions',{}).get(photo_value,'Photo')
    show_photo(int(photo_value),photo_caption)
