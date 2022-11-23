
import tempfile
import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt

if 'xgboost_hog' not in st.session_state:
    st.session_state.xgboost_hog = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-hog.pkl", 'rb'))
if 'xgboost_raw' not in st.session_state:
    st.session_state.xgboost_raw = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-raw.pkl", 'rb'))
if 'xgboost_pr' not in st.session_state:
    st.session_state.xgboost_pr = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-pr.pkl", 'rb'))
if 'ss_hog' not in st.session_state:
    st.session_state.ss_hog = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-hog.pkl", 'rb'))
if 'ss_raw' not in st.session_state:
    st.session_state.ss_raw = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-raw.pkl", 'rb'))
if 'ss_pr' not in st.session_state:
    st.session_state.ss_pr = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-pr.pkl", 'rb'))

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


modelname = st.sidebar.selectbox(
    'Please select a model?',
    ('XGBoost', "SVM"))

if modelname == modelname == 'XGBoost':
    preprocess = st.sidebar.selectbox(
        'Please select a feature extraction method?',
        ('Projections', 'Raw images', "histogram of gradient"))
else:
    preprocess = st.sidebar.selectbox(
        'Please select a feature extraction method?',
        ('Raw images', "countours"))
if st.sidebar.button("Run"):
    if(modelname == 'XGBoost' and preprocess == 'histogram of gradient'):
        model = st.session_state['xgboost_hog']
        ss = st.session_state['ss_hog']
        img = cv2.resize(cv2.imread(name, 0), (100, 33))
        features, vis = hog(np.array(img), block_norm='L2-Hys',visualize=True)
        st.write("Feature Map")
        st.write(vis.shape)
        st.image(vis/255, width=500)
        prob = model.predict_proba(ss.transform(features.reshape(1, features.shape[0])))
        st.write('probability')
        st.write("predicted class: ", np.argmax(prob), ", probability: ",np.max(prob))
    if(modelname == 'XGBoost' and preprocess == 'Raw images'):
        model = st.session_state['xgboost_raw']
        ss = st.session_state['ss_raw']
        img = cv2.resize(cv2.imread(name, 0), (100, 33)).flatten()
        prob = model.predict_proba(ss.transform(img.reshape(1, img.shape[0])))
        st.write('probability')
        st.write("predicted class: ", np.argmax(prob), ", probability: ",np.max(prob))
    if(modelname == 'XGBoost' and preprocess == 'Projections'):
        model = st.session_state['xgboost_pr']
        ss = st.session_state['ss_pr']
        img = np.sum(cv2.resize(cv2.imread(name, 0), (100, 33)), axis=0)
        st.write(img.shape)
        fig, x = plt.subplots()
        x.plot(img)
        st.pyplot(fig)
        prob = model.predict_proba(ss.transform(img.reshape(1, img.shape[0])))
        st.write('probability')
        st.write("predicted class: ", np.argmax(prob), ", probability: ",np.max(prob))
