#importing the necessary libraries
from flask import Flask,render_template,url_for,request,jsonify
from flask_restful import reqparse, abort, Api, Resource
import urllib.request, json
import os
import warnings
import string, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import *
import tokenizers

tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file = 'vocab-roberta-base.json',
    merges_file = 'merges-roberta-base.txt',
    lowercase = True,
    add_prefix_space = True
)

MAX_LEN = 384

def build_model():
  ids = tf.keras.layers.Input((MAX_LEN,), dtype = tf.int32)
  att = tf.keras.layers.Input((MAX_LEN,), dtype = tf.int32)
  tok = tf.keras.layers.Input((MAX_LEN,), dtype = tf.int32)

  config = RobertaConfig.from_pretrained('./config-roberta-base.json')
  bert_model = TFRobertaModel.from_pretrained('./pretrained-roberta-base.h5', config = config)
  x = bert_model(ids, attention_mask=att, token_type_ids = tok)

  # For start logit

  x1 = tf.keras.layers.Dropout(0.1)(x[0])
  x1 = tf.keras.layers.Conv1D(1,1)(x1)
  x1 = tf.keras.layers.Flatten()(x1)
  x1 = tf.keras.layers.Activation('softmax')(x1)

  # For end logit

  x2 = tf.keras.layers.Dropout(0.1)(x[0])
  x2 = tf.keras.layers.Conv1D(1,1)(x2)
  x2 = tf.keras.layers.Flatten()(x2)
  x2 = tf.keras.layers.Activation('softmax')(x2)

  # Initalising the model

  model = tf.keras.models.Model(inputs = [ids, att, tok], outputs = [x1, x2])
  optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5)
  model.compile(loss='categorical_crossentropy', optimizer = optimizer)

  return model


def generate_ans(inp,que,model,tokenizer):
  inp_id = np.zeros((1,MAX_LEN),dtype='int32')
  attn_mask_input = np.zeros((1,MAX_LEN),dtype='int32')
  token_type_id_input = np.zeros((1,MAX_LEN),dtype='int32')
  inpenc = tokenizer.encode(inp)
  queenc = tokenizer.encode(que)
  inp_id[0,:len(inpenc.ids)+len(queenc.ids) + 4] = [0] + queenc.ids + [2,2] + inpenc.ids + [2]
  attn_mask_input[0,:len(inpenc.ids)+len(queenc.ids) + 4] = 1
  s, f = model.predict([inp_id,attn_mask_input,token_type_id_input])
  s_ = np.argmax(s[0,])
  f_ = np.argmax(f[0,])
  ans = tokenizer.decode(inpenc.ids[s_ - 1: f_ + 1])

  return ans




model = build_model()
model.load_weights('./weights.h5')


app = Flask(__name__)
@app.route('/')
def main():
    return render_template('main.html')

#For displaying predicted value
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action = 'ignore',category = FutureWarning)

    if request.method == 'POST':
        message1 = request.form['message1']
        message2 = request.form['message2']
        my_prediction = generate_ans(inp = message1, que = message2,model = model,tokenizer = tokenizer)
    return render_template('main.html',prediction = my_prediction)


if __name__ == '__main__':
  app.run(debug = True, use_reloader = False)
