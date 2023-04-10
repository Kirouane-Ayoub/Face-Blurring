import cv2
import mediapipe as mp
import cvzone
import time
import streamlit as st

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
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


    col1 , _, col2 = st.columns(3)

    with col1 : 
        st.write("Click To start Detection")
        bstart = st.button("Start")
    with col2 : 
        if save == "Yes" :     
            st.write("Double Click to save")
            bstop = st.button("Save")
        elif save == "No" : 
            st.write("Click To Stop")
            bstop = st.button("Stop")

    if bstart : 
        cap = cv2.VideoCapture(source)
        if save == "Yes" : 
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
            out = cv2.VideoWriter(f'results/{str(time.asctime())}.mp4',
                                    fourcc, 10, (w, h)) 

        frame_window = st.image([])
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for id, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, ic = image.shape
                        bbox = int(bboxC.xmin * iw -20), int(bboxC.ymin * ih-20), \
                            int(bboxC.width * iw+20), int(bboxC.height * ih+20)
                        cvzone.cornerRect(image ,(bbox) ,l=10)
                        x , y , h , w = int(bboxC.xmin * iw-20) , int(bboxC.ymin * ih-20) ,int(bboxC.height * ih + 20) ,int(bboxC.width * iw +20)
                        crop = image[y:y+h , x:x+w] 
                        blur = cv2.blur(crop , (50,50))
                        image[y:y+h , x:x+w] = blur
                        try : 
                            out.write(image)
                        except : 
                            pass

                img = cv2.cvtColor( image , cv2.COLOR_BGR2RGB)
                frame_window.image(img)

        if bstop : 
            cap.release()