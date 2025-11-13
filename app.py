import streamlit as st
import numpy as np
import cv2
import os
import urllib.request

# Tắt log TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# Custom functions
@register_keras_serializable()
def signed_sqrt(x): return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x) + 1e-9)
@register_keras_serializable()
def square_fn(x): return tf.math.square(x)
@register_keras_serializable()
def scaling(x, scale=0.17): return x * scale

# PWA
st.set_page_config(page_title="Kinship AI", layout="centered")
st.markdown('<link rel="manifest" href="/static/manifest.json">', unsafe_allow_html=True)

# Tải model từ Release
@st.cache_resource
def get_model():
    path = "model.keras"
    if not os.path.exists(path):
        url = "https://github.com/le198/kinship-recognition-pwa/releases/download/v1.0/facenet_vgg.keras"
        with st.spinner("Tải model..."):
            urllib.request.urlretrieve(url, path)
    try:
        model = load_model(path, custom_objects={
            'signed_sqrt': signed_sqrt,
            'square_fn': square_fn,
            'scaling': scaling
        }, compile=False)
        return model
    except:
        st.error("Lỗi load model")
        return None

model = get_model()
if not model:
    st.stop()

# Face detection
@st.cache_resource
def get_net():
    proto = "face_detector/deploy.prototxt"
    weights = "face_detector/model.caffemodel"
    os.makedirs("face_detector", exist_ok=True)
    if not os.path.exists(proto):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            proto
        )
    if not os.path.exists(weights):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            weights
        )
    return cv2.dnn.readNetFromCaffe(proto, weights)

net = get_net()

def crop_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    for i in range(dets.shape[2]):
        conf = dets[0,0,i,2]
        if conf > 0.5:
            box = dets[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2 = box.astype(int)
            face = img[y1:y2,x1:x2]
            if face.size > 0:
                return cv2.resize(face, (224,224))
    return cv2.resize(img, (224,224))

# UI
st.title("Kinship Recognition")
c1, c2 = st.columns(2)

with c1:
    f1 = st.file_uploader("Ảnh 1", ["jpg","png"])
    if f1:
        img1 = cv2.imdecode(np.frombuffer(f1.read(), np.uint8), 1)
        face1 = crop_face(img1)
        st.image(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))

with c2:
    f2 = st.file_uploader("Ảnh 2", ["jpg","png"])
    if f2:
        img2 = cv2.imdecode(np.frombuffer(f2.read(), np.uint8), 1)
        face2 = crop_face(img2)
        st.image(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))

if f1 and f2:
    with st.spinner("Đang so sánh..."):
        x1 = np.expand_dims(face1 / 255.0, 0)
        x2 = np.expand_dims(face2 / 255.0, 0)
        pred = model.predict([x1,x2,x1,x2], verbose=0)
        score = float(pred[0][0])
        thresh = st.slider("Ngưỡng", 0.5, 1.0, 0.9, 0.01)
        st.metric("Score", f"{score:.4f}")
        if score > thresh:
            st.success("**CÓ QUAN HỆ**")
        else:
            st.warning("**KHÔNG CÓ QUAN HỆ**")
