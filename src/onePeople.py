
"""## Setup"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers, metrics, Model

#vggface
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import cv2
from os import listdir
from os.path import  join
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


image_size = 224
batch_size = 3
from random import randint
import matplotlib.image as mpimg
from PIL import Image

class Reader:
    def __init__(self, dir_images):
        self.root = dir_images
        self.list_classes = os.listdir(self.root)
        self.not_single = [c for c in self.list_classes if len(listdir(join(self.root, c)))>1]
        self.list_classes_idx = range(len(self.list_classes))
        self.not_single_idx = range(len(self.not_single))

        self.weights_not_single = [len(listdir(join(self.root, c))) for c in self.not_single]
        self.weights_not_single = np.array(self.weights_not_single)
        self.weights_not_single = self.weights_not_single / np.sum(self.weights_not_single)

        self.weights = [len(listdir(join(self.root, c))) for c in self.list_classes]
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)

    def GetTriplet(self):
        # positive and anchor classes are selected from folders where have more than two pictures
        idx_class_pos = np.random.choice(self.not_single_idx, 1 ,p=self.weights_not_single)[0]
        name_pos = self.not_single[idx_class_pos]
        class_pos = join(self.root, name_pos)
        dir_pos = os.listdir( class_pos )
        pos_num=len(dir_pos)
        idx_img_anchor = randint(0,pos_num-1)
        idx_img_pos = randint(0,pos_num-1)
        while idx_img_pos == idx_img_anchor:
          idx_img_pos = randint(0,pos_num-1)
        path_anchor = join(class_pos, dir_pos[idx_img_anchor])
        path_pos = join(class_pos, dir_pos[idx_img_pos])

        # negative classes are selected from all folders
        while True:
            idx_class_neg = np.random.choice(self.list_classes_idx, 1, p=self.weights)[0]
            if idx_class_neg != idx_class_pos:
                break
        name_neg = self.list_classes[idx_class_neg]
        class_neg = join(self.root, name_neg)
        dir_neg = os.listdir( class_neg )
        neg_num = len(dir_neg)
        idx_img_neg = randint(0,neg_num-1)
        path_neg = join(class_neg, dir_neg[idx_img_neg])

        return path_anchor, path_pos, path_neg

def _ReadAndResize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((image_size, image_size))
    im = np.array(im, dtype="float32")
    im = im / 255.0 #test
    return im

def _Flip(im_array):
    if np.random.uniform(0, 1) > 0.7:
        im_array = np.fliplr(im_array)
    return im_array

# create triplet example from LFW dataset
def TripletGenerator(reader, test=False):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(batch_size):
            path_anchor, path_pos, path_neg = reader.GetTriplet()
            img_anchor = _Flip(_ReadAndResize(path_anchor))
            img_pos = _Flip(_ReadAndResize(path_pos))
            img_neg = _Flip(_ReadAndResize(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        image_count = len(list_pos)
        print(image_count)
        anchor_dataset = tf.data.Dataset.from_tensor_slices(list_anchor)
        positive_dataset = tf.data.Dataset.from_tensor_slices(list_pos)
        negative_dataset = tf.data.Dataset.from_tensor_slices(list_neg)
        return  anchor_dataset, positive_dataset, negative_dataset

train = Reader(dir_images='Train/')

train_anchor_dataset, train_positive_dataset, train_negative_dataset = TripletGenerator(train)
train_dataset = tf.data.Dataset.zip((train_anchor_dataset, train_positive_dataset, train_negative_dataset))
train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)


from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model
from keras import backend as K

# create a vggface model
base_model = VGGFace(weights='vggface', include_top=False, input_shape=(224, 224, 3))
# base_model.summary()

base_model.trainable = False
target_shape = (224, 224)

from tensorflow import keras
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
flatten = layers.Flatten()(x)

dense1 = layers.Dense(256, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
output = layers.Dense(64)(dense1)

embedding = Model(base_model.input, output, name="Embedding")

one_input = layers.Input(shape=target_shape + (3,))
one_embedding = embedding(one_input)

siamese_network = Model(inputs=one_input, outputs=one_embedding)

class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        anchor_embedding = self.siamese_network(data[0])
        positive_embedding = self.siamese_network(data[1])
        negative_embedding = self.siamese_network(data[2])

        ap_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        an_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), -1)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

# 实例化模型并编译
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))

# 创建一个迭代器来获取数据集中的一个样本
sample_data = next(iter(train_dataset))

# 使用一个样本来调用模型
siamese_model(sample_data[0])


from keras import optimizers, Input, Model, layers
from keras.layers import Dense, Flatten, Lambda, GlobalAveragePooling2D

margin = 0.7
alpha = 1.0

def contrastive_loss(y_true, y_pred):
  return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def euclidean_dist(inputs):
  assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)

  u, v = inputs
  return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))

# Create VGGFace base model
base_model = VGGFace(weights='vggface', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Flatten(name='flatten')(x)
x = Dense(256, name='fc6')(x)
x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_normalize')(x)
x = Dense(64)(x)

# Embedding model
embedding_model = Model(inputs=base_model.input, outputs=x)

# Input layers
inp_1 = Input(shape=(224, 224, 3), name='one')
inp_2 = Input(shape=(224, 224, 3), name='two')

# Output layers
out_1 = embedding_model(inp_1)
out_2 = embedding_model(inp_2)

# Euclidean distance
merge_layer = Lambda(euclidean_dist)([out_1, out_2])

# Multi-output model
Contras_model = Model(inputs=[inp_1, inp_2], outputs=[merge_layer, out_1, out_2])

# Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
Contras_model.compile(loss=[contrastive_loss, None, None], optimizer=optimizer)

import warnings
warnings.filterwarnings("ignore")

import sys
sys.argv=['']
del sys


def _get_image_batches(batch_size, list_name, img_size):
  while True:
    X1 = []
    X2 = []
    y = []

    for _ in range(batch_size):

      choice = random.choice(['same', 'different'])

      if choice == 'same':
        person_1 = random.choice(list_name)
        person_2 = person_1
        y.append(0.)

      else:
        person_1, person_2 = random.sample(list_name, 2)
        y.append(1.)

      image_files_1 = glob.glob(person_1+"/*.jpg")
      image_files_2 = glob.glob(person_2+"/*.jpg")

      img_fi_1 = random.choice(image_files_1)
      img_fi_2 = random.choice(image_files_2)

      face_1 = cv2.imread(img_fi_1)
      face_2 = cv2.imread(img_fi_2)

      face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2RGB)
      face_2 = cv2.cvtColor(face_2, cv2.COLOR_BGR2RGB)

      face_1 = cv2.resize(face_1, img_size, cv2.INTER_AREA)
      face_2 = cv2.resize(face_2, img_size, cv2.INTER_AREA)

      if list_name == train_people:
        face_1 = augment_image(face_1)
        face_2 = augment_image(face_2)

      face_1 = face_1 / 255.
      face_2 = face_2 / 255.

      X1.append(face_1)
      X2.append(face_2)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    y = np.asarray(y)

    yield [X1, X2], y
    

def step_decay(epoch):

  initial_lrate = 0.01
  drop = 0.1
  epochs_drop = 1.0

  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

  return lrate


def augment_image(img_new):
  gamma = random.choice([i for i in np.arange(1.0, 2.5, 0.25)])
  img_new = img_new / 255.
  img_new = img_new ** (1/gamma)
  img_new *= 255
  img_new = img_new.astype('uint8')

  augmentation = np.random.choice(['Flip', 'Rotation', 'None'], p=[0.2, 0.6, 0.2])

  if augmentation == 'Flip':
    img_new = np.fliplr(img_new)

  elif augmentation == 'Rotation':
    angle = random.choice([i for i in range(-5, 5)])

    rows, cols, channels = img_new.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_new = cv2.warpAffine(img_new, M, (cols, rows))

  else:
    img_new = img_new

  return img_new


from numpy import asarray
from numpy import expand_dims
import pickle

def load_and_process_image(image_path):
  # load the photo
  face = mpimg.imread(image_path)

  image = Image.fromarray(face)
  image = image.resize((224, 224))
  pixels = asarray(image)
  # convert one face into samples
  pixels = pixels.astype('float32')
  samples = expand_dims(pixels, axis=0)
  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=2)

  return samples

def tri_face_recognition(image_path):
  image = load_and_process_image(image_path)
  pred = siamese_model.predict(image)

  return pred


def ssl_face_recognition(image_path1, image_path2):
  image1 = load_and_process_image(image_path1)
  image2 = load_and_process_image(image_path2)

  # perform prediction
  pred = Contras_model.predict([image1, image2])   # yhat = vggface.predict(samples) vggface是上面第一個儲存格提到

  return pred[1], pred[2]


def compute_euclidean_distance(embedding1, embedding2):
  distance = tf.reduce_sum(tf.square(embedding1 - embedding2), -1)  # 原本計算距離的公式
  return distance


def load_model_weights(model, weight_path):
    model.load_weights(weight_path)
    return model


people_num = 20    # 總共有多少位學生
model_num = 10     # 總共有幾個模型
trip_model_num = 5   # 三重有多少個模型

modelPath = "weight/"  # 模型權重放的位置
# train_folder = 'Train/'  # Train路徑
goal_image_path = 'cut_photo.jpg'  # 捕獲的照片路徑


pre_outcome = {}        # 紀錄個模型的投票結果
for model_index in range (1, model_num+1): # 初始化
    pre_outcome[model_index] = 0  # 驗證集


for model_index in range (1, model_num+1) :
  modelName = "model_" + str(model_index) + ".h5"    # model名稱!!!!!!!!!!!!!!!!!!!!!!!!

  # ==================================================== Triplet Network ====================================================
  if model_index < trip_model_num+1 :  # 要看前幾個是tri

    # load model
    siamese_model.load_weights(modelPath + modelName)   # 載入模型權重

    filename = "embedding" + str(model_index) + ".pkl"
    with open(filename, "rb") as f:
      train_embeddings = pickle.load(f)

    # ------------------ 得到教授的embeddings，並計算距離 ------------------

    # predict
    embedding = tri_face_recognition(goal_image_path)    # 計算這張照片和Train的哪張照片最距離最近
    min_distance = float('inf')
    closest_train_subfolder = 0

    for train_subfolder, train_embedding in train_embeddings.items():
        distance = compute_euclidean_distance(embedding, train_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_train_subfolder = train_subfolder
    # print(f"vModel {model_index}: Predict: {closest_train_subfolder}" )
    pre_outcome[model_index] = closest_train_subfolder

  # ==================================================== Contrastive Network ====================================================
  else :
    # load model
    Contras_model.load_weights(modelPath + modelName)    # 載入模型權重

    filename = "embedding" + str(model_index) + ".pkl"
    with open(filename, "rb") as f:
      train_embeddings = pickle.load(f)


    # ------------------ 得到教授的embeddings，並計算距離 ------------------

    embedding1, embedding2 = ssl_face_recognition(goal_image_path, goal_image_path)
    # 計算這張照片和Train的哪張照片最距離最近
    min_distance = float('inf')
    closest_train_subfolder = 0

    for train_subfolder, train_embedding in train_embeddings.items():
        distance = compute_euclidean_distance(embedding1, train_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_train_subfolder = train_subfolder
    # print(f"vModel{model_index} Predict: {closest_train_subfolder}" )
    pre_outcome[model_index] = closest_train_subfolder


from collections import Counter
# print( "\n「教授你好!!」\n" )
# 使用 Counter 進行計數
count = Counter(pre_outcome.values())

# 找到最常出現的數字
most_common_number = count.most_common(1)[0][0]
# print(f"Ensemble Model: Most Common Predict: {most_common_number}")

# 將預測結果返回json格式數據
import json
result = {
    "face_prediction": most_common_number
}

print("<!--START-->")
print(json.dumps(result))
print("<!--END-->")