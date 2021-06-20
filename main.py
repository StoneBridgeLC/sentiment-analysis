import pickle
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
max_len = 40
with open('tokenizer.pickle', 'rb') as myfile:
    tokenizer = pickle.load(myfile)
loaded_model = load_model('best_transfer_model.h5')

def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if score > 0.5:
    return 1
  else:
    return 0

import requests
import json
import os
import time
import logging

# set logger
LOG_FORMAT = "%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# get dataserver domain
#dataserverDomain = os.environ['DATASERVER_DOMAIN']
dataserverDomain = 'http://localhost:8080'

# define job : request and response
def job():
  while True:
    # GET
    res = requests.get(dataserverDomain + '/comment/unlabeled?limit=100')
    if res.status_code != 200:
      logger.error('failed to requests.get : {}'.format(res.text))
      return

    count = 0
    resBody = res.json()
    for item in resBody:
      count += 1
      item['is_pos'] = sentiment_predict(item['body'])
      logger.info('{label} {body}'.format(label=item['is_pos'], body=item['body']))

    if count == 0:
        return

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.put(dataserverDomain + '/comment/label', data=json.dumps(resBody), headers=headers)
    if res.status_code != 200:
      logger.error('failed to requests.put : {}'.format(res.text))
      return


while True:
  job()
  time.sleep(60 * 10)

