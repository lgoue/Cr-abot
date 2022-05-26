# Creabot

## TO DO
yaml files
## Getting started

- download spacy model with topic env activate:
`python -m spacy download en_core_web_sm`

## Launch conversation

### Valence arousal video server
- change Furhat IP and dev IP in vaAPI.json
- `python vaAPI.py`

### Topic management server
- on another process :
- `conda activate topic`
- `python topicAPI.py`

## Eval server
- on another process :
- `conda activate topic`
- `python evalAPI.py`

## Eval usefulness server
- on another process :
- `conda activate bert`
- `python evalBertAPI.py`


- start the skill on furhat
