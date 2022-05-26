from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
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
from transformers import BertModel
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertModel

data = pd.read_csv("s1_data_long.csv",encoding="ISO-8859-1",sep=";",decimal=",")

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H,H1, D_out = 768,168,250 ,5

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(H,H1),
            nn.LeakyReLU(),
            nn.Linear(H1,D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        #last_hidden_state_cls = torch.concat(outputs[1])

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return torch.softmax(logits,dim=1)
# Create the BertClassfier class
class BertClassifier3(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier3, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H,H1, D_out = 768,168,250 ,5

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(H,H1),
            nn.LeakyReLU(),
            nn.Linear(H1,D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        #last_hidden_state_cls = torch.concat(outputs[1])

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return torch.softmax(logits,dim=1)
class BertClassifier2(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier2, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H,H1, D_out = 768*2,168,250 ,4

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_item = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(H,H1),
            nn.LeakyReLU(),
            nn.Linear(H1,D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert_item.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask,input_ids_item, attention_mask_item):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        outputs_item = self.bert_item(input_ids=input_ids_item,
                            attention_mask=attention_mask_item)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls =torch.concat((outputs[0][:, 0, :],outputs_item[0][:, 0, :]),dim=-1)
        #last_hidden_state_cls = torch.concat(outputs[1])

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return torch.softmax(logits,dim=1)

# Create the BertClassfier class
class BertClassifierNovelty(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=True):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifierNovelty, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H,H1, D_out = 768*2,168,250 ,4

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pool = nn.AvgPool1d(768*2)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask,input_ids_item, attention_mask_item):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        outputs_item = self.bert(input_ids=input_ids_item,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = torch.concat((outputs[1],outputs_item[1]),dim=-1)
        #last_hidden_state_cls = torch.concat(outputs[1])

        # Feed input to classifier to compute logits
        logits = self.pool(last_hidden_state_cls)

        return logits


from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,       # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
MAX_LEN = 25
# function to remove the filler words
import spacy
#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
sw_spacy.add('use')
sw_spacy.add('make')
sw_spacy.add('build')
print(sw_spacy)

def remove_fill(text):
  words = [word for word in text.split() if word.lower() not in sw_spacy]
  new_text = " ".join(words)
  return new_text

items=['box','rope']

model = {}
for i in items:
  m = BertClassifier(freeze_bert=False)
  m.load_state_dict(torch.load("./models/model_useful_"+i+".pt"))
  model[i] = m
  model[i].eval()

model_feasability = {}
for i in items:
  m = BertClassifier3(freeze_bert=False)
  m.load_state_dict(torch.load("./models/model_appropriatness_"+i+".pt",map_location=torch.device('cpu')))
  model_feasability[i] = m
  model_feasability[i].eval()


import gensim
from gsdmm import MovieGroupProcess
 # Load data and set labels
gsdmm_box = MovieGroupProcess(K=15, alpha=0.8, beta=0.8, n_iters=30)
gsdmm_box=gsdmm_box.from_data(15, 0.8, 0.8, 1282, 13, np.load('models/doc_count_box.npy',allow_pickle=True), np.load('models/cluster_word_count_box.npy',allow_pickle=True), np.load('models/cluster_word_distribution_box.npy',allow_pickle=True))
gsdmm_rope = MovieGroupProcess(K=15, alpha=0.8, beta=0.8, n_iters=30)
gsdmm_rope=gsdmm_rope.from_data(15, 0.8, 0.8, 1052, 13, np.load('models/doc_count_rope.npy',allow_pickle=True), np.load('models/cluster_word_count_rope.npy',allow_pickle=True), np.load('models/cluster_word_distribution_rope.npy',allow_pickle=True))

gsdmms = {}
gsdmms['box'] = gsdmm_box
gsdmms['rope'] = gsdmm_rope
model_crea = BertClassifierNovelty(freeze_bert=True)
#model_save_name = 'creativity_score.pt'
#path = F"./models/{model_save_name}"
#model_crea.load_state_dict(torch.load(path))
model_crea.eval()
@app.get('/get_ellaboration_score')
def get_idea_ellaboration_score(idea:str):
      idea = idea.replace("_", " ")
      t = remove_fill(idea)
      t= t.split(" ")
      return len(t)
@app.get('/get_usefulness_score')
def get_usefulness_score(idea: str,item:str):
      idea = idea.replace("_", " ")
      input, mask = preprocessing_for_bert([remove_fill(idea)])
      logits = np.argmax(model[item](input, mask).detach().cpu().numpy()[0])
      return str(logits)

@app.get('/get_feasability_score')
def get_feasability_score(idea: str,item:str):
      idea = idea.replace("_", " ")
      input, mask = preprocessing_for_bert([remove_fill(idea)])
      logits = np.argmax(model_feasability[item](input, mask).detach().cpu().numpy()[0])
      return str(logits)

@app.get('/get_creativity_score')
def get_creativity_score(idea: str,item:str):
      idea = idea.replace("_", " ")
      input, mask = preprocessing_for_bert([remove_fill(idea)])
      input_item, mask_item=preprocessing_for_bert([item])
      logits = model_crea(input, mask,input_item, mask_item).detach().cpu().numpy()[0][0]
      return str(logits)

@app.get('/get_novelty_score')
def get_nov_score(idea: str,item:str):
      idea = idea.replace("_", " ")
      scores = gsdmms[item].score(remove_fill(idea))
      c=0
      for i,p in enumerate(scores):
          c += p/gsdmms[item].cluster_doc_count[i]

      if item=='box':
          s=round((c-0.013)*2000)
      else:
          s=round((c-0.0156)*2500)
      s = min(4,max(s,0))

      return str(s)

import uvicorn
print("launching app")
uvicorn.run(app, port=8088,host=config['Dev_IP'])
