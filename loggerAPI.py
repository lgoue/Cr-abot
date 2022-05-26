
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import numpy as np

# Load configuration
with open('vaAPI.json') as f:
  config = json.load(f)
print(config)

log_directory ="log/"
pomdp_url = "http://"+config['Dev_IP']+":8888/"
app = FastAPI()
v=[]
a=[]
hap = []
sad = []
ne = []
path ="na"
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.type = "00"
cols=["condition","n_turn","strat","da","p","a","item","use","novelty","ellaboration","usefulness","feasability","hap","sadness","angry","surprise","neutral","attendfurhat","attendpad","attendupleft","mutualgaze","smile","valance_video","arousal_video","happy_audio","sad_audio","neutral_audio"]
for m in range(3):
    for i in range(5):
        cols.append('belief_m_'+str(m)+'_i_'+str(i))
print(len(cols))
@app.get('/get_path')
def get_path():
    global path
    return path
@app.get('/new_user')
def new_user(name_log:str):
    global v
    global a
    global hap
    global sad
    global ne
    global path
    path=name_log
    df = pd.DataFrame(columns=cols)
    df.to_csv(log_directory+name_log,index=False)
    result = requests.get(
    pomdp_url+"new_interaction"
)
    v = []
    a = []
    hap = []
    sad = []
    ne = []
    return True

@app.get('/update_va')
def update_va(val:str,ar:str):
    global v
    global a
    v.append(float(val))
    a.append(float(ar))
    return True
@app.get('/update_audio_emo')
def update_va(h:str,s:str,n:str):
    global hap
    global sad
    global ne
    hap.append(float(h))
    sad.append(float(s))
    ne.append(float(n))
    return True
@app.get('/get_v_video')
def get_v_video():
    global v
    global a
    if len(v)>0:
        return np.mean(v)
    else :
        return  -1
@app.get('/get_hap_audio')
def get_hap_audio():
    global hap
    if len(hap)>0:
        return np.mean(hap)
    else :
        return  -1
@app.get('/get_sad_audio')
def get_sad_audio():
    global sad
    if len(sad)>0:
        return np.mean(sad)
    else :
        return  -1
@app.get('/get_ne_audio')
def get_ne_audio():
    global ne
    if len(ne)>0:
        return np.mean(ne)
    else :
        return -1
@app.get('/get_a_video')
def get_a_video():
    global v
    global a
    if len(a)>0:
        return np.mean(a)
    else :
        return -1
@app.get('/log')
def log(turn_log : str,name_log:str):
    global v
    global a
    global hap
    global sad
    global ne
    df = pd.read_csv(log_directory+name_log)
    row = turn_log.split(',')
    if len(v)>0:
        row+=[np.mean(v),np.mean(a)]
        v=[]
        a=[]
    else :
        row+=[-1,-1]
    if len(hap)>0:
        row+=[np.mean(hap),np.mean(sad),np.mean(ne)]
        hap=[]
        sad=[]
        ne=[]
    else :
        row+=[-1,-1,-1]
    for m in range(3):
        for i in range(5):
            result = requests.get(
            pomdp_url+"get_belief_state",
            params ={'m': str(m), 'i':str(i)}
            )
            row += [result.text]
    print(row)

    df=df.append(pd.DataFrame([row],columns=cols),ignore_index=True)

    df.to_csv(log_directory+name_log,index=False)
    result = requests.get(
    pomdp_url+"save",
    params ={'log_file': name_log}
    )
    return True

import uvicorn
print("launching app")
uvicorn.run(app, port=8008,host=config['Dev_IP'])
