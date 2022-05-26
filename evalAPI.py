
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
app = FastAPI()
# Load configuration
with open('vaAPI.json') as f:
  config = json.load(f)
print(config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


import torch
import torch.nn as nn
import re
import numpy as np
from bertopic import BERTopic
from flair.data import Sentence
# function to remove the filler words
import spacy
#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
sw_spacy.add('use')
sw_spacy.add('build')
sw_spacy.add('make')
print(sw_spacy)
glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])
document_glove_embeddings.load_state_dict(torch.load("models/glove_embeddings"))

def remove_fill(text):
  words = [word for word in text.split() if word.lower() not in sw_spacy]
  new_text = " ".join(words)
  return new_text

items=['box','rope']

Ns = {}
counts = {}
model = {}
for i in items:
  m = BERTopic.load("models/"+i+"_novelty")
  m.calculate_probabilities=False
  m2 = BERTopic.load("models/"+i+"_novelty_2")
  m2.calculate_probabilities=False
  counts[i] = [np.array(m.get_topic_info().Count),np.array(m2.get_topic_info().Count)]
  model[i] = [m,m2]



x={}
maxs = {}
for i in items:

  max0 = np.max(counts[i][0][1:])
  max1 = np.max(counts[i][1][1:])
  min1 = np.min(counts[i][1][1:])
  max1 = np.max(counts[i][1][1:])
  min0 = np.min(counts[i][0][1:])
  if i=='rope':
    min1 = np.min(counts[i][1])
    max1 = np.max(counts[i][1])
  min = np.min([min0,min1])
  max = np.max([max0,max1])
  maxs[i] = max
  x[i] = max * 5 * min/(4*max-min)



@app.get('/get_novelty_score')
def get_idea_novelty_score(idea:str,item:str):
      idea = idea.replace("_", " ")
      s = Sentence(idea)
      document_glove_embeddings.embed(s)
      emb = np.array(s.embedding.unsqueeze(0))
      t = model[item][0].transform([remove_fill(idea)],  embeddings=emb)[0][0]
      if t==-1:
        t = model[item][1].transform([remove_fill(idea)])[0][0]
        if item =='rope':
            freq = 1/counts[item][1][t]
        else:
            freq = 1/counts[item][1][t+1]
        if t==-1:
            return 5
      else:
          freq = 1/counts[item][0][t+1]
      return round(x[i]*(freq-(1/maxs[i])))

@app.get('/get_ellaboration_score')
def get_idea_ellaboration_score(idea:str):
      idea = idea.replace("_", " ")
      t = remove_fill(idea)
      t= t.split(" ")
      return len(t)
import pandas as pd
def get_idea_novelty_score_item(idea:str,item='na'):
      item='box'
      idea = idea.replace("_", " ")
      s = Sentence(idea)
      document_glove_embeddings.embed(s)
      emb = np.array(s.embedding.unsqueeze(0))
      t = model[item][0].transform([remove_fill(idea)],  embeddings=emb)[0][0]
      if t==-1:
        #t = model[item][1].transform([remove_fill(idea)])[0][0]
        #if item =='rope':
        #    freq = 1/counts[item][1][t]
        #else:
        #    freq = 1/counts[item][1][t+1]
        if t==-1:
            return 5
      else:
          freq = 1/counts[item][0][t+1]
      return freq
data = pd.read_csv("rope_data.csv",sep=';')
data_rope = data.sample(1000)
data_rope['novelty'] = data_rope['response'].apply(lambda x : get_idea_novelty_score_item(x,'rope'))
data_rope.to_csv('rope_data_test.csv')
#data = pd.read_csv("rope_data.csv",sep=';')
#data_rope = data
#data_rope['novelty'] = data_rope['response'].apply(lambda x : get_idea_novelty_score_item(x,'rope'))
#data_rope.to_csv('rope_data.csv')
import uvicorn

import uvicorn
print("launching app")
uvicorn.run(app, port=8800,host=config['Dev_IP'])
