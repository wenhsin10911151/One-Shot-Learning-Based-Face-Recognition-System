
"""## Setup"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model


import cv2
import os
from os import listdir
from os.path import  join
import numpy as np
import mediapipe as mp

# 拍照
image_path = 'photo.jpg'  # ********************* 捕獲的照片路徑 *************************

def save_faces(image_path):
    img = cv2.imread(image_path)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Convert the image to RGB (MediaPipe face detection requires RGB images)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                sub_face = img[y:y + h, x:x + w]

                if sub_face.size == 0:
                    continue  # Skip empty sub_face

                resized_image = cv2.resize(sub_face, (224, 224))

                # Create save folder if it doesn't exist
                save_folder_path = 'cut_photo.jpg'  # ****************************放裁切完的路徑*********************
                cv2.imwrite(os.path.join(save_folder_path), resized_image)

save_faces(image_path)
