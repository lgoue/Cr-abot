
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load configuration
with open('vaAPI.json') as f:
  config = json.load(f)
print(config)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# function to remove the filler words
import spacy
#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
sw_spacy.add('use')
sw_spacy.add('build')
sw_spacy.add('make')
print(sw_spacy)

def remove_fill(text):
  words = [word for word in text.split() if word.lower() not in sw_spacy]
  new_text = " ".join(words)
  return new_text

import pandas as pd
import numpy as np
from bertopic import BERTopic


items=['box','rope']

names = {}
proba = {}

names_grouped = {}
proba_grouped = {}
model = {}
defs = {}
for i in items:

  data = np.load("models/"+i+"_topics.npy")
  m = BERTopic.load("models/"+i+"_topics")
  names[i] = []
  proba[i] = []
  model[i] = m
  for b in data:
   names[i].append(b[0])
   proba[i].append((float(b[1])+1)**2)
   defs[b[0]] = b[2]
  proba[i] = np.array(proba[i]) / np.sum(proba[i])

  df = pd.DataFrame(np.array([names[i],proba[i].astype(np.float)]).T,columns=['topic','proba'])
  df['proba'] = df['proba'].astype(np.float)
  df = df.groupby('topic').sum()
  names_grouped[i] =np.array(df['proba'].index)
  proba_grouped[i] =df['proba'].to_numpy().astype(np.float)

@app.get('/topic_def')
def topic_def(topic: str):
    return defs[topic]

@app.get('/idea_topic')
def idea_topic(idea: str,item:str):
    idea = idea.replace("_"," ")
    return names[item][model[item].transform([remove_fill(idea)])[0][0]]

@app.get('/next_topic')
def next_topic(proposed_topics : str,item:str):
    proposed_topics = proposed_topics[1:-1]
    proposed_topics = proposed_topics.split(', ')
    topics = []
    topics_proba = []
    for i,t in enumerate(names_grouped[item][1:]):
      if t not in proposed_topics:
        topics += [t]
        topics_proba+= [proba_grouped[item][i+1]]
      topics_proba = [p/np.sum(topics_proba) for p in topics_proba]
    if len(topics)>0:
      draw = np.random.choice(topics, 1,p=topics_proba)

      return draw[0]
    else:
      return "STOP"


import uvicorn
print("launching app")
uvicorn.run(app, port=8000,host=config['Dev_IP'])
