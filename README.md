# Face Detection in Python Using  Webcam

“Computer vision and machine learning have really started to take off, but for most people, the whole idea of what a computer is seeing when it’s looking at an image is relatively obscure.” – Mike Kreiger

 visit [AI PROJECTS](https://aihubprojects.com) for more tutorials.
Following are the requirements for it:-

  1. Python 2.7/3
  2. OpenCV3/4
  3. Numpy
  4. Haar Cascade Frontal face classifiers

In this tutorial we are going to build a model which detects the face in video from webcam in realtime. we are building the project on python3.7 , opencv4 & Numpy. But you can run the code in any version on python and opencv. Using Haar Cascade classifier it first detects your face and draw a rectangle aaround it to indicate your face. Make sure you have webcam installed in your computer. Press 'q' to exit the demo.

Lets dive into the code:

OpenCV comes with a trainer as well as detector. Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc. Those XML files are stored in the opencv/data/haarcascades/ folder. Let's create a  real time face with OpenCV.

first we need to import requisites and required XML classifiers. 

```python
import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```

This line sets the video source to the default webcam, which OpenCV can easily capture.

```python
video_capture = cv2.VideoCapture(0)
```

NOTE: You can also provide a filename here, and Python will read in the video file. However, you need to have ffmpeg installed for that since OpenCV itself cannot decode compressed video. Ffmpeg acts as the front end for OpenCV, and, ideally, it should be compiled directly into OpenCV. This is not easy to do, especially on Windows.

```python
while True:
    
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags=cv2.CV_HAAR_SCALE_IMAGE
    )

    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
```

Here, we capture the video. The read() function reads one frame from the video source, which in this example is the webcam. This returns:

  1. The actual video frame read (one frame on each loop)
  2. A return code
The return code tells us if we have run out of frames, which will happen if we are reading from a file. This doesn’t matter when reading from the webcam, since we can record forever, so we will ignore it.

We wait for the ‘q’ key to be pressed. If it is, we exit the script.

Testing the application
Now it’s time to test our application by executing the following command in the terminal:

```python
python <python script name>.py
```

Once the script gets executed successfully, you will be able to see yourself in the frame and a rectangle is drawn around your face as shown in below image.
  
That’s all. Hope you got idea on real time face detection in webcam using Python 3.

You may be interested to read see yourself in webcam using Python

Thanks for reading.
