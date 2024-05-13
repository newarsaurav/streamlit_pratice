# to run this 
# go to terminal and run streamlit run steam_app.py
# or streamlit run "15.Model Deployment\streamlit Deployment\steamlit_app.py"  

import streamlit as st
from ultralytics import YOLO 
from PIL import Image

model = YOLO('yolov8n.pt')
with st.sidebar:
    thresh = st.slider('Threshold', min_value=0.20 , max_value=0.99 , value=0.5)

with st.expander('about this app '):
    st.text('This app was created in class')
    
img_file = st.file_uploader('Bruv upload your image', type=['png' , 'jpg'], help='This should only be images my bruv.')
if img_file:
    c1 , c2 = st.columns(2)
    # st.image(img_file , caption='This is your uploaded image.')
    c1.image(img_file , caption='This is your uploaded image.' , use_column_width=True)    
    Image.open(img_file).save(img_file.name)
    results = model(img_file.name , stream=False , conf = thresh)
    results[0].save(filename = 'result.jpg')
    # st.image('result.jpg' , caption='This is your predicted')
    c2.image('result.jpg'  , caption='This is your predicted image.' , use_column_width=True)

st.camera_input('camera', key=None, help=None, on_change=None, args=None, kwargs=None, label_visibility="visible")