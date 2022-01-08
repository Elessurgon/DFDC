
import tempfile
import cv2 as cv
from numpy import asarray
import numpy as np
from PIL import Image
import streamlit as st
from backend import predict_on_video_set
import base64



st.set_page_config(page_title='DFDC', layout = 'wide', initial_sidebar_state = 'auto')

st.markdown(
    """
    <style>
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        padding-left: 75px !important;
        float: right;
        margin-right: 300px;
        margin-left: -100px;
    }
    .stApp {
        # background-color: #EA5455;
        # background-color: aliceblue;
        # background-image: linear-gradient(to bottom right, #FEB692, #EA5455);
        background-image: linear-gradient(to bottom right, #ABCDFF, #0396FF);
    }
    .css-18e3th9 {
        width: 90%;
        padding: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" width="140px" height="140px" src="data:image/png;base64,{base64.b64encode(open("assets/logo_0.png", "rb").read()).decode()}">
        <h1 class="logo-text"">RV College of Engineering<br><small>(Autonomous Institution Affiliated to Visvesvaraya Technological University, Belagavi)</small></h1>
    </div>
    <div style="margin-top: -25px;">
      <small></small><br><small>
                DEPARTMENT OF MASTER OF COMPUTER APPLICATIONS<br>
                Bengaluru– 560059</small>
    </div><hr>

    """,
    unsafe_allow_html=True
)


st.header("""Deepfakes Video detection app""")


f = st.file_uploader("Upload file",  type=['mp4'])

if f:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    vf = cv.VideoCapture(tfile.name)
    video_file = open(tfile.name, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    ans = predict_on_video_set([tfile.name], num_workers=4)

    if (ans[0] - 0.5 < 1e-6):
        st.success(f"The video is a deepfake with a probability of {ans[0]:.2f}")
    else:
        st.error(f"The video is a deepfake with a probability of {ans[0]:.2f}")