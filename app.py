import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from keras.utils import load_img, img_to_array
from PIL import Image, ImageEnhance

st.title("Ocular Disease Detection System (ODDS)")
st.header("By  Priya  Thiagarajan  (21CS007)")

files = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

#if (st.button("Submit")): 
  
  #basepath = os.path.dirname(__file__)
  #filename = files.filename
  #filename = filename.replace(" ", "")
  #file_path = os.path.join(basepath, 'uploads', filename)
  #files.save(file_path)

model_name = 'ODDS_Model_2.h5'
model = load_model(model_name)

if (st.button("Submit")):
  #img = image.load_img(file_path, target_size=[240, 240])
  img = Image.open(files)
  
  x = img_to_array(img)
  x = x.resize(300, 300)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
#   classes = model.predict(images, batch_size=10)
  classes = model.predict(images)
  # print (classes)
  x = np.argmax(classes)
  
#   img = img_to_array(img)
#   img = img.resize(240, 240)
#   img = np.expand_dims(img, axis=0)
  
#   images = np.vstack([img])
#   x = model.predict(images, batch_size=10)

  # op = model.predict_classes(img)
  # x = np.argmax(op)
  
  if x == 0:
    st.success("a")
  if x == 1:
    st.warning("b")
  if x == 2:
    st.error("c")
  if x == 2:
    st.error("d")
