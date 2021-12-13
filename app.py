
import tempfile
import cv2 as cv
from numpy import asarray
import numpy as np
from PIL import Image
import streamlit as st
from backend import predict_on_video_set


st.write("""Deepfakes Video detection app""")


f = st.file_uploader("Upload file",  type=['mp4'])

if f:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    vf = cv.VideoCapture(tfile.name)
    video_file = open(tfile.name, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    ans = predict_on_video_set(
        [tfile.name], num_workers=4)

    if (ans[0] - 0.5 < 1e-6):
        st.success(
            f"The video is a deepfake with a probability of {ans[0]:.2f}")
    else:
        st.error(f"The video is a deepfake with a probability of {ans[0]:.2f}")
