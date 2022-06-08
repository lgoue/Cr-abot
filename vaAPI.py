import numpy as np
from pathlib import Path
import argparse
import cv2



import requests
import zmq
import time
import json


def detect(net, img, confidence_threshold):
    #detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392,(256,256),(0,0,0),True,crop=False)
    print(blob.shape)
    with torch.no_grad():
        out = net(torch.tensor(blob).float())
    val = out['valence']
    ar = out['arousal']

    return {'val':val,'ar':ar}


# Load configuration
with open('vaAPI.json') as f:
  config = json.load(f)
print(config)



torch.backends.cudnn.benchmark =  True

device='cpu'

state_dict_path = Path(__file__).parent.joinpath('emonet/pretrained', f'emonet_8.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=8).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

# Setup the sockets
context = zmq.Context()

# Input camera feed from furhat using a SUB socket
insocket = context.socket(zmq.SUB)
insocket.setsockopt_string(zmq.SUBSCRIBE, '')
insocket.connect('tcp://' + config["Furhat_IP"] + ':3000')
insocket.setsockopt(zmq.RCVHWM, 1)
insocket.setsockopt(zmq.CONFLATE, 1)  # Only read the last message to avoid lagging behind the stream.

# Output results using a PUB socket
context2 = zmq.Context()
outsocket = context2.socket(zmq.PUB)
outsocket.bind("tcp://" + config["Dev_IP"] + ":" + config["detection_exposure_port"])

print('connected, entering loop')
prevset = {}
iterations = 1
detection_period = config["detection_period"] # Detecting objects is resource intensive, so we try to avoid detecting objects in every frame
detection_threshold = config["detection_confidence_threshold"] # Detection threshold takes a double between 0.0 and 1.0
x = True
# choose codec according to format needed
v,a = 0,0
h = -1
logger_url = "http://192.168.137.1:8008/"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 1, (256, 256))
while True:

    string = insocket.recv()
    magicnumber = string[0:3]
    if magicnumber == b'\xff\xd8\xff':
        buf = np.frombuffer(string,dtype=np.uint8)
        img = cv2.imdecode(buf,flags=1)
        if h > 0:
            img = img[y:y+h, x:x+w]

        height, width, layers = img.shape
        size = (width,height)
        video.write(img)

        if (iterations % detection_period == 0):
            print("Detecting VA!")
            buf = np.frombuffer(string,dtype=np.uint8)
            img = cv2.imdecode(buf,flags=1)
            height,width,channels = img.shape
            res = detect(net,img, detection_threshold)
            print(res)
            msg = 'val_'+str(res['val'].numpy()[0])
            msg+=' ar_'+str(res['ar'].numpy()[0])
            result = requests.get(
            logger_url+"update_va",
            params ={'val': (res['val'].numpy()[0]), 'ar': res['ar'].numpy()[0]}
            )


        iterations = iterations + 1
    else :
        json_object = json.loads(string)
        h = -1
        try :
            print(json_object['users'][0]['emotion'])
            x = json_object['users'][0]['bbox']['x']
            y = json_object['users'][0]['bbox']['y']
            w = json_object['users'][0]['bbox']['w']
            h = json_object['users'][0]['bbox']['h']
        except:
            pass


    k = cv2.waitKey(1)
    if k%256 == 27: # When pressing esc the program stops.
        # ESC pressed
        print("Escape hit, closing...")
        break
