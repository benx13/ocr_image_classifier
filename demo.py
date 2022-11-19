
import tempfile
import streamlit as st
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from torch import nn
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'resnet18' not in st.session_state:
    resnet18 =  torchvision.models.resnet18(pretrained=False)
    resnet18.fc = nn.Linear(512, 22)
    resnet18 = nn.DataParallel(resnet18)
    resnet18.load_state_dict(torch.load('models/resnet18_100acc_224.pt'))
    resnet18.to(device)
    st.session_state.resnet18 = resnet18
    
if 'efficientNetB0' not in st.session_state:
    efficientNetB0 =  torchvision.models.efficientnet_b0(pretrained=False)
    efficientNetB0.classifier[1] = nn.Linear(1280, 22)
    efficientNetB0 = nn.DataParallel(efficientNetB0)
    efficientNetB0.load_state_dict(torch.load('/home/benx13/code/ocr/ocr_image_classifier/models/efficientNetB0_100acc_224.pt'))
    efficientNetB0.to(device)
    st.session_state.efficientNetB0 = efficientNetB0   


st.set_page_config(layout='wide')
st.title('Words classifier(works on gpu only)')

file = st.file_uploader('Choose an image')
if (file is not None):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    name = tfile.name
    st.write('please wait reading image ...')

else:
    name = '/home/benx13/code/ocr/ocr_image_classifier/data_2021/data_2021/png/ce79_040.png'
    st.write('please wait reading default image ...')
st.image(cv2.imread(name, 0), width=500)


if st.button('EfficientNet'):
    with torch.no_grad():
        model = st.session_state['efficientNetB0'].to(device).eval()
        transform = T.Compose([T.Resize((224, 224)),T.ToTensor()]) 
        img = Image.open(name)
        tensor = transform(img).to(device).unsqueeze(0)
        prob = F.softmax(model(tensor)).cpu().detach().numpy()
        dict = {str(i):prob[0,i] for i in range(22)}
        st.write("softmax probability normalization:")
        st.write(dict)
        st.write("predicted class: ", np.argmax(prob), ", probability: ",np.max(prob))

if st.button('Restnet18'):
    with torch.no_grad():
        model = st.session_state['resnet18'].to(device).eval()
        transform = T.Compose([T.Resize((224, 224)),T.ToTensor()]) 
        img = Image.open(name)
        tensor = transform(img).to(device).unsqueeze(0)
        prob = (F.softmax(model(tensor)).cpu().detach().numpy())
        dict = {str(i):prob[0,i] for i in range(22)}
        st.write("softmax probability normalization:")
        st.write(dict)
        st.write("predicted class: ", np.argmax(prob), ", probability: ",np.max(prob))




