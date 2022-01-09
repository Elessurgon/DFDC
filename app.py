
import tempfile
import cv2 as cv
from numpy import asarray
import numpy as np
from PIL import Image
import streamlit as st
from backend import predict_on_video_set
import base64

st.set_page_config(page_title='DFDC', layout='wide',
                   initial_sidebar_state='auto')

st.markdown(
    """
    <style>
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        float: right;
        margin-right: 13rem;
    }
    .stApp {
        # background-color: #EA5455;
        # background-color: aliceblue;
        # background-image: linear-gradient(to bottom right, #FEB692, #EA5455);
        background-image: linear-gradient(to bottom right, #ABCDFF, #0396FF);
    }
    .css-18e3th9 {
        width: 70%;
        padding: 2rem;
        text-align: center;
    }
    .st-bs {
        background-color: #ff000066;
    }
    .st-bv {
        background-color: #09ab3b69;
    }
    .st-bu .st-br {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" width="140px" height="140px" src="data:image/png;base64,{base64.b64encode(open("assets/logo_0.png", "rb").read()).decode()}">
        <h1 class="logo-text"">RV College of Engineering<br><h5><small>(Autonomous Institution Affiliated to Visvesvaraya Technological University, Belagavi)</small></h5></h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("DEPARTMENT OF MASTER OF COMPUTER APPLICATIONS  \nBengaluru- 560059")


st.markdown("""<hr>""",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
    <div style="text-align: right;">
        <h5 style="color: #000000">Project By:</h5>
    </div>
    """,
        unsafe_allow_html=True
    )

with col2:
    st.write("M Shamanth  \n1RV20MC038")

with col3:
    st.markdown(
        f"""
    <div style="text-align: left;">
        Mathias Russel Rudolf Richard<br>1RV20MC047
    </div>
    """,
        unsafe_allow_html=True
    )
    st.write("")


col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
    <div style="text-align: right;">
        <h5 style="color: #000000">Under the Guidance of:</h5>
    </div>
    """,
        unsafe_allow_html=True
    )
with col2:

    st.write("Dr. Vijayalakshmi M.N  \nAssociate Professor  \nDepartment of MCA  \nRV College of Engineering®  \nBengaluru-560059")


st.markdown(
    f"""<hr>""",
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
        st.success(
            f"The video is a deepfake with a probability of {ans[0]:.2f}")
    else:
        st.error(f"The video is a deepfake with a probability of {ans[0]:.2f}")
