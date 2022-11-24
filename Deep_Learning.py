
import tempfile
import streamlit as st
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from torch import nn
import torch
import torchvision.transforms as T
from PIL import Image

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
d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'Resnet18' not in st.session_state:
    resnet18 =  torchvision.models.resnet18(pretrained=False)
    resnet18.fc = nn.Linear(512, 22)
    resnet18 = nn.DataParallel(resnet18)
    resnet18.load_state_dict(torch.load('models/resnet18_100acc_224.pt'))
    resnet18.to(device)
    st.session_state.resnet18 = resnet18
    
if 'EfficientNetB0' not in st.session_state:
    efficientNetB0 =  torchvision.models.efficientnet_b0(pretrained=False)
    efficientNetB0.classifier[1] = nn.Linear(1280, 22)
    efficientNetB0 = nn.DataParallel(efficientNetB0)
    efficientNetB0.load_state_dict(torch.load('/home/benx13/code/ocr/ocr_image_classifier/models/efficientNetB0_100acc_224.pt'))
    efficientNetB0.to(device)
    st.session_state.efficientNetB0 = efficientNetB0   


st.set_page_config(layout='wide')
st.title('Words classifier Deep Learning')

file = st.file_uploader('Choose an image')
if (file is not None):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    name = tfile.name
    # st.write('please wait reading image ...')

else:
    name = '/home/benx13/code/ocr/ocr_image_classifier/data_2021/data_2021/png/bf38_039.png'
    #st.write('please wait reading default image ...')
st.image(cv2.imread(name, 0), width=500)

option = st.sidebar.selectbox(
    'Please select a model',
    ('EfficientNetB0', 'Restnet18'))

if st.sidebar.button("Run"):

    if option == 'EfficientNetB0':
        with torch.no_grad():
            model = st.session_state['efficientNetB0'].to(device).eval()
            transform = T.Compose([T.Resize((224, 224)),T.ToTensor()]) 
            img = Image.open(name)
            tensor = transform(img).to(device).unsqueeze(0)
            prob = F.softmax(model(tensor)).cpu().detach().numpy()
            dict = {str(i):prob[0,i] for i in range(22)}
            col1, col2 = st.columns(2)
            with col1:
                st.write("Softmax probability normalization:")
                st.write(dict)
            with col2:
                st.write("Predicted class: ", np.argmax(prob), ", Word: ", d[np.argmax(prob)], ", Probability: ", np.max(prob))


    if option == 'Restnet18':
        with torch.no_grad():
            model = st.session_state['resnet18'].to(device).eval()
            transform = T.Compose([T.Resize((224, 224)),T.ToTensor()]) 
            img = Image.open(name)
            tensor = transform(img).to(device).unsqueeze(0)
            prob = (F.softmax(model(tensor)).cpu().detach().numpy())
            dict = {str(i):prob[0,i] for i in range(22)}
            col1, col2 = st.columns(2)
            with col1:
                st.write("softmax probability normalization:")
                st.write(dict)
            with col2:
                st.write("predicted class: ", np.argmax(prob), ", Word: ", d[np.argmax(prob)], ", probability: ",np.max(prob))




