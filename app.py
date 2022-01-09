
import tempfile
import cv2 as cv
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
    .css-17z2rne  {
        color: #000000;
    }
    .link {
        text-decoration: none !important;
        color: #fff !important;
        border-radius: 0.25rem;
        padding: 5px;
        background-color: black;
    }
    a.link:hover {
        color: #f00 !important;
        border: 1px solid #f00;
    }
    .intro {
        text-align: justify;
        color: black;
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

if 'show' not in st.session_state:
	st.session_state.show = False

placeholder = st.empty()

if not st.session_state.show:
    with placeholder.container():
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

        st.header("""Deepfake Video Detection Application""")
        st.markdown(
            f"""<div class='intro'><p>An application of deep learning in recent times is generation of fake videos. According to Forbes, Generative Adversarial Networks(GANs) generated fake videos are growing exponentially every year.</p>
            <p>The project proposed to use deep learning to counter this and find videos which have synthesized faces. This project aimed to investigate fake videos and to detect them by using deep learning detection techniques specialized in object detection, classification in videos.</p></div>
            <br>""",
            unsafe_allow_html=True
        )

click=False
button = st.empty()
if not st.session_state.show:
    with button.container():
        click = st.button("Open Application")

if click:
    st.session_state.show = True
    placeholder.empty()
    button.empty()

if st.session_state.show:
    st.markdown('<a class="link" target="_self" href="https://share.streamlit.io/elessurgon/dfdc/app.py"><b>Go Home</b></a>', unsafe_allow_html=True)

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
