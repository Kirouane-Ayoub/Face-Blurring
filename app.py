import cv2 
import streamlit as st
from cvzone.FaceDetectionModule import FaceDetector
import time 
tab0 , tab1 = st.tabs(["Home" , "Detection"])

with st.sidebar:
    st.image("icon.png")
    detect_from = st.selectbox("Detect Faces from : " ,
                                ["File" , "Live"])
    save = st.radio("Do you want to save results ? " , 
                    ("Yes" , "No"))


with tab0:
    st.header("About This Project : ")
    st.image("blur.jpg")
    st.write("""Face blur is a technique used to protect the privacy of individuals in images or videos. 
    It involves blurring or obscuring the facial region of a person to prevent their identification. 
    Face blur is commonly used in situations where individuals may have a reasonable expectation of privacy, 
    such as in CCTV footage, social media posts, or news broadcasts. The process of face blur can be performed manually or 
    using computer vision algorithms. Manual face blurring involves manually selecting and blurring the face region, while computer
    vision algorithms use techniques such as face detection and recognition to automatically blur faces in images or videos.
    Face blur is an important area of research in computer vision and has significant applications in maintaining the privacy of individuals.
""")



with tab1 : 
    if detect_from == "File" : 
        source = st.file_uploader("Upload Ur Video : " ,
                                   type=["mp4" , "mvi"])
        if source : 
            source = source.name
    else : 
        LIVE = st.selectbox("Select Live type : " , 
                            ["WebCam" , "URL"])
        if LIVE == "WebCam" :
            source = st.selectbox("Select Your Index Device : " , 
                                  (1,2,3))

        else : 
            source = st.text_input("Entre Your Url here :")


    start , _, stop = st.columns(3)

    with start : 
        st.write("Click To start Detection")
        bstart = st.button("Start")
    with stop : 
        if save == "Yes" :     
            st.write("Double Click to save")
            bstop = st.button("Save")
        elif save == "No" : 
            st.write("Click To Stop")
            bstop = st.button("Stop")

    if bstart : 
        try : 
            cap = cv2.VideoCapture(source)
            if save == "Yes" : 
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
                out = cv2.VideoWriter(f'results/{str(time.asctime())}.mp4',
                                    fourcc, 10, (w, h)) 
            frame_window = st.image([])
            detector = FaceDetector(minDetectionCon=0.75)
            while 1: 
                _ , img = cap.read()
                img , bboxes = detector.findFaces(img , draw=True)
                if bboxes : 
                    for bbox in bboxes : 
                        x,y,w,h = bbox["bbox"]
                        if x  < 0 : x = 0
                        if y  < 0 : y = 0
                        crop = img[y:y+h , x:x+w]
                        blur = cv2.blur(crop , (50,50))
                        img[y:y+h , x:x+w] = blur
                        try : 
                            out.write(img)
                        except : 
                            pass
                img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
                frame_window.image(img)
        except :
            pass
        if bstop : 
            cap.release()
