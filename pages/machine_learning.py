
import tempfile
import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog

if 'xgboost' not in st.session_state:
    st.session_state.xgboost = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-hog.pkl", 'rb'))

st.set_page_config(layout='wide')
st.title('Words classifier Machine Learning')

file = st.file_uploader('Choose an image')
if (file is not None):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    name = tfile.name
    st.write('please wait reading image ...')

else:
    name = '/home/benx13/code/ocr/ocr_image_classifier/data_2021/data_2021/png/bf38_039.png'
    st.write('please wait reading default image ...')
st.image(cv2.imread(name, 0), width=500)



preprocess = st.sidebar.selectbox(
    'Please select a feature extraction method?',
    ('Projections', 'Raw images', "histogram of gradient"))
modelname = st.sidebar.selectbox(
    'Please select a model?',
    ('XGBoost', "SVM"))
if st.sidebar.button("Run"):
    model = st.session_state['xgboost']
    img = cv2.resize(cv2.imread(name, 0), (100, 33)).flatten()
    st.write(img.shape)
    features = hog(np.array(img), block_norm='L2-Hys')
    st.write(features.shape)