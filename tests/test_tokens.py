from app.tokens import render, presence_template, sanitize_name, summarise_group

PEOPLE={1:{'person_id':1,'display_name':None,'merged_into':None,'deleted':False},7:{'person_id':7,'display_name':None,'merged_into':None,'deleted':False},9:{'person_id':9,'display_name':'Riya','merged_into':7,'deleted':False}}
def test_default_labels(): assert render('{{p:1}} and {{p:7}}',PEOPLE)=='User 1 and User 7'
def test_rename(): PEOPLE[1]['display_name']='Aryan'; assert render('{{p:1}} swims',PEOPLE)=='Aryan swims'; PEOPLE[1]['display_name']=None
def test_merge_redirect(): assert render('{{p:9}} waves',PEOPLE)=='User 7 waves'
def test_pending_face(): assert render('{{f:88}} smiles',PEOPLE)=='an unidentified person smiles'
def test_sanitize(): assert '{{' not in sanitize_name('{{p:3}} <script>')
def test_presence(): assert presence_template(['{{p:1}}','{{p:7}}'])=='{{p:1}} and {{p:7}} are present in this image.'
def test_group(): assert summarise_group(['A','B','C','D','E','F'])=='A, B, C and 3 others'
