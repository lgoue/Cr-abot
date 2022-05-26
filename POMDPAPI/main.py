import numpy as np
from action import ActionType
from interaction import InteractionModel
from utils import entropy
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from observationfurhat import MoodObservation
from observation import Idea
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

parser = argparse.ArgumentParser()
parser.add_argument("--condition")
parser.add_argument("--load")
logger_url = "http://192.168.137.1:8008/"
# Get the hyperparameters
args_parse = parser.parse_args()
cond=args_parse.condition
args={
    "reward_crea":6,
    "epsilon":0.2,
    "condition":cond,
    "gamma":0.2,
    "wp":0.8
}

load = args_parse.load
env = InteractionModel(cond,config =args,load=load)


@app.get('/get_action')
def get_action():

    action = env.get_action()
    print(action.to_string())
    return action.to_string()

@app.get('/update_state')
def update_state(idea_score:str,p:str,a:str,d:str,hap:str,sad:str,ang:str,sur:str,ne:str):
#    mood_observation = MoodObservation(float(p),float(a))
    hap_audio =float( requests.get(
            logger_url+"get_hap_audio",
            params ={}
            ).text)
    sad_audio = float( requests.get(
            logger_url+"get_sad_audio",
            params ={}
            ).text)
    ne_audio = float( requests.get(
            logger_url+"get_ne_audio",
            params ={}
            ).text)
    if hap_audio > 0:
        if hap*sad*ne > 0:
            mood_observation = MoodObservation((float(hap)+hap_audio)/2,(float(sad)+sad_audio)/2,float(ang),float(sur),(float(ne)+ ne_audio)/2)
        else :
            mood_observation = MoodObservation((float(hap_audio)),(float(sad_audio)),float(ang),float(sur),(float(ne_audio)))
    else:
        mood_observation = MoodObservation((float(hap)),(float(sad)),float(ang),float(sur),(float(ne)))
    idea_observation = Idea(int(idea_score))
    env.update_state(mood_observation,idea_observation)
    env.update_action(env.state)
    return True

@app.get('/get_belief_state')
def get_belief_state(m:str,i:str):
    print("hello")
    state = env.get_belief_state()
    return float(state[int(m),int(i)])

@app.get('/save')
def save(log_file:str):
    env.agent.save('save_models/'+log_file+"_")
    env.agent.save('save_models/')
    return True

@app.get('/new_interaction')
def new_interaction():
    env.new_interaction()
    try:
        env.agent.load('save_models/')
    except:
        print("first user")
    env.new_interaction()
    env.update_action(env.state)
    return True

import uvicorn
print("launching app")
uvicorn.run(app, port=8888,host="192.168.137.1")
