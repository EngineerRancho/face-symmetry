# face-symmetry
The Face Symmetry Checker is a Python script that utilizes facial landmark detection to analyze the symmetry of a person's face. It uses the dlib library for face detection and facial landmark prediction.
Features

Detects facial landmarks using the dlib library.
Measures the symmetry of various facial features, including the chin to ear, lip corner to eye and ear, nose to ear, and forehead.
Provides symmetry percentages for each facial feature, indicating the level of asymmetry between the left and right sides of the face.
Generates visual representations of facial features with symmetry lines overlaid on the input image.
Supports real-time face detection and symmetry analysis from live camera feed or pre-captured images.


# Installation

    git clone https://github.com/EngineerRancho/face-symmetry.git
    cd face-symmetry
    git clone https://github.com/fenollp/data.shape_predictor_68_face_landmarks
    cd data.shape_predictor_68_face_landmarks
    mv shape_predictor_68_face_landmarks.dat ..
    pip install -r requirements.txt
    python symmetry.py

# Usage

Run the script and provide the path to the image containing the face to be analyzed.
The script will detect facial landmarks and calculate symmetry percentages for each facial feature.
It will display the input image with symmetry lines overlaid, highlighting the analyzed facial features.
Symmetry percentages for each feature will be printed to the console, indicating the level of symmetry or asymmetry between the left and right sides of the face.

# Requirements

    Python 3.x
    dlib library
    OpenCV (cv2)



   ## The dlib library for providing facial landmark detection capabilities.
   ## OpenCV for image processing and visualization.
