import streamlit as st  

st.set_page_config(layout='wide')
st.title('Words classifier')

image = st.file_uploader('Choose an image')
st.image(image, width=500)

model = st.selectbox(
    'Select model',
    ('Model1', 'Model2', 'Model3'))

extraction = st.selectbox(
    'Select an extraction model',
    ('Type1', 'Type2', 'Type3'))

def predict(image, model, extraction):
    return 'Class i'

predicted = predict(image, model, extraction)

if st.button('Run'):
    st.write('The class is ' + predicted)