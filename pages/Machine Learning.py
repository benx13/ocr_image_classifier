
import tempfile
import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt

d = {i:'' for i in range(22)}
d[0] = 'المنزه 9'
d[1] = 'المرناقيّة 20 مارس'
d[2] = 'سيدي إبراهيم الزهّار'
d[3] = 'المنزه 6'
d[4] = 'غنّوش'
d[5] = 'شتاوة صحراوي'
d[6] = 'رأس الذّراع'
d[7] = 'سبعة آبار'
d[8] = 'شعّال'
d[9] = 'الخليج'
d[10] = 'تطاوين 7 نوفمبر'
d[11] = 'أآودة'
d[12] = 'شمّاخ'
d[13] = 'نقّة'
d[14] = 'المحارزة 18'
d[15] = 'نّحال'
d[16] = 'مارث'
d[17] = 'الدخّانية'
d[18] = 'تلّ الغزلان'
d[19] = 'سيدي الظّاهر'
d[20] = 'الفايض'
d[21] = 'الرضّاع'



def contours_extract(img):
    h,w=img.shape
    img = cv2.threshold(img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contour = np.zeros(img.shape)
    for y in range(h-1):
        for x in range(w-1):
            if img[y,x] != img[y+1,x]:
                contour[y,x]=255
            if img[y,x] != img[y,x+1]:
                contour[y,x]=255
    return contour.reshape(-1)

if 'xgboost_hog' not in st.session_state:
    st.session_state.xgboost_hog = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-hog.pkl", 'rb'))
if 'xgboost_raw' not in st.session_state:
    st.session_state.xgboost_raw = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-raw.pkl", 'rb'))
if 'xgboost_pr' not in st.session_state:
    st.session_state.xgboost_pr = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/xgboost-pr.pkl", 'rb'))
if 'svm_raw' not in st.session_state:
    st.session_state.svm_raw = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/svm-raw.pkl", 'rb'))
if 'svm_countours' not in st.session_state:
    st.session_state.svm_countours = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/svm-countours.pkl", 'rb'))
if 'ss_hog' not in st.session_state:
    st.session_state.ss_hog = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-hog.pkl", 'rb'))
if 'ss_raw' not in st.session_state:
    st.session_state.ss_raw = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-raw.pkl", 'rb'))
if 'ss_pr' not in st.session_state:
    st.session_state.ss_pr = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-pr.pkl", 'rb'))
if 'ss_countours' not in st.session_state:
    st.session_state.ss_countours = pickle.load(open("/home/benx13/code/ocr/ocr_image_classifier/models/ss-countours.pkl", 'rb'))

st.set_page_config(layout='wide')
st.title('Words classifier Machine Learning')

file = st.file_uploader('Choose an image')
if (file is not None):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    name = tfile.name
    #st.write('please wait reading image ...')

else:
    name = '/home/benx13/code/ocr/ocr_image_classifier/data_2021/data_2021/png/bf38_039.png'
    #st.write('please wait reading default image ...')
st.image(cv2.imread(name, 0), width=500)

modelname = st.sidebar.selectbox(
    'Please select a model?',
    ('XGBoost', "SVM"))

if modelname == modelname == 'XGBoost':
    preprocess = st.sidebar.selectbox(
        'Please select a feature extraction method?',
        ('Projections', 'Raw images', "Histogram of gradient"))
else:
    preprocess = st.sidebar.selectbox(
        'Please select a feature extraction method?',
        ('Raw images', "countours"))
if st.sidebar.button("Run"):
    if(modelname == 'XGBoost' and preprocess == 'Histogram of gradient'):
        model = st.session_state['xgboost_hog']
        ss = st.session_state['ss_hog']
        img = cv2.resize(cv2.imread(name, 0), (100, 33))
        features, vis = hog(np.array(img), block_norm='L2-Hys',visualize=True)
        st.write("Feature Map")
        st.image(vis/255, width=500)
        prob = model.predict_proba(ss.transform(features.reshape(1, features.shape[0])))
        #st.write('probability')
        st.write("Predicted class: ", np.argmax(prob), ", word: ", d[np.max(prob)], ", probability: ", np.max(prob))
    if(modelname == 'XGBoost' and preprocess == 'Raw images'):
        model = st.session_state['xgboost_raw']
        ss = st.session_state['ss_raw']
        img = cv2.resize(cv2.imread(name, 0), (100, 33)).flatten()
        prob = model.predict_proba(ss.transform(img.reshape(1, img.shape[0])))
        #st.write('probability')
        st.write("Predicted class: ", np.argmax(prob), ", word: ", d[np.max(prob)], ", probability: ", np.max(prob))
    if(modelname == 'XGBoost' and preprocess == 'Projections'):
        model = st.session_state['xgboost_pr']
        ss = st.session_state['ss_pr']
        img = np.sum(cv2.resize(cv2.imread(name, 0), (100, 33)), axis=0)
        fig, x = plt.subplots() 
        fig.set_size_inches(6, 3)
        x.plot(img)
        st.pyplot(fig)
        prob = model.predict_proba(ss.transform(img.reshape(1, img.shape[0])))
        #st.write('probability')
        st.write("Predicted class: ", np.argmax(prob), ", word: ", d[np.max(prob)], ", probability: ", np.max(prob))
    if(modelname == 'SVM' and preprocess == 'countours'):
        model = st.session_state['svm_countours']
        ss = st.session_state['ss_countours']
        img = contours_extract(cv2.resize(cv2.imread(name, 0), (100, 33)))
        st.image(img.reshape((33,100))/255, width=500)
        prob = model.predict(ss.transform(img.reshape(1, img.shape[0])))
        #st.write('predicted class: ', str(prob[0]))
        st.write("Predicted class: ", np.argmax(prob), ", word: ", d[np.max(prob)], ", probability: ", np.max(prob))
    if(modelname == 'SVM' and preprocess == 'Raw images'):
        model = st.session_state['svm_raw']
        ss = st.session_state['ss_raw']
        img = cv2.resize(cv2.imread(name, 0), (100, 33)).flatten()
        prob = model.predict(ss.transform(img.reshape(1, img.shape[0])))
        #st.write('predicted class: ', str(prob[0]))
        st.write("Predicted class: ", np.argmax(prob), ", word: ", d[np.max(prob)], ", probability: ", np.max(prob))
