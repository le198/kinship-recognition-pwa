import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ====================== TỐI ƯU RAM ======================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log TF
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import math

# ====================== CUSTOM FUNCTIONS ======================
@register_keras_serializable(package="Custom")
def signed_sqrt(tensor):
    return tf.math.sign(tensor) * tf.math.sqrt(tf.math.abs(tensor) + 1e-9)

@register_keras_serializable(package="Custom")
def square_fn(t):
    return tf.math.square(t)

@register_keras_serializable(package="Custom")
def scaling(x, scale=0.17):
    return x * scale

def patch_square(x): return tf.math.square(x)
def patch_signed_sqrt(x): return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x) + 1e-9)

# ====================== PWA ======================
st.set_page_config(page_title="Kinship AI", layout="centered")
st.markdown('<link rel="manifest" href="/static/manifest.json">', unsafe_allow_html=True)

# ====================== DOWNLOAD MODEL NHẸ ======================
@st.cache_resource
def load_kinship_model():
    model_path = "facenet_vgg.keras"
    
    if not os.path.exists(model_path):
        url = "https://github.com/le198/kinship-recognition-pwa/releases/download/v1.0/facenet_vgg.keras"
        with st.spinner("Tải model (~1-2 phút lần đầu)..."):
            try:
                urllib.request.urlretrieve(url, model_path)
                st.success("Model tải xong!")
            except:
                st.error("Không tải được model.")
                return None

    try:
        with st.spinner("Khởi động AI..."):
            model = load_model(
                model_path,
                custom_objects={
                    'signed_sqrt': signed_sqrt,
                    'square_fn': square_fn,
                    'scaling': scaling,
                    'tf': tf,
                    'math': math
                },
                safe_mode=False,
                compile=False
            )
            # Patch Lambda
            for layer in model.layers:
                if 'Lambda' in layer.__class__.__name__:
                    name = layer.name.lower()
                    if 'square' in name or 'lambda_2' in name:
                        layer.function = patch_square
                    elif 'signed_sqrt' in name:
                        layer.function = patch_signed_sqrt
            return model
    except Exception as e:
        st.error("Lỗi load model.")
        st.code(str(e))
        return None

model = load_kinship_model()
if not model:
    st.stop()

# ====================== FACE DETECTION ======================
@st.cache_resource
def load_face_net():
    path = "face_detector"
    os.makedirs(path, exist_ok=True)
    proto = f"{path}/deploy.prototxt"
    weights = f"{path}/res10_300x300_ssd_iter_140000.caffemodel"
    
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

net = load_face_net()

def crop_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = img[y1:y2, x1:x2]
            if face.size > 0:
                return cv2.resize(face, (224, 224))
    return cv2.resize(img, (224, 224))

# ====================== UI ======================
st.title("Kinship Recognition")
col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Ảnh 1", type=["jpg", "png"], key="1")
    if img1_file:
        img1 = cv2.imdecode(np.frombuffer(img1_file.getvalue(), np.uint8), 1)
        face1 = crop_face(img1)
        st.image(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB), "Face 1")

with col2:
    img2_file = st.file_uploader("Ảnh 2", type=["jpg", "png"], key="2")
    if img2_file:
        img2 = cv2.imdecode(np.frombuffer(img2_file.getvalue(), np.uint8), 1)
        face2 = crop_face(img2)
        st.image(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB), "Face 2")

if img1_file and img2_file:
    with st.spinner("So sánh..."):
        x1 = np.expand_dims(face1.astype(np.float32), 0)
        x2 = np.expand_dims(face2.astype(np.float32), 0)
        
        # Giả lập preprocess (bạn có thể thêm đúng hàm nếu cần)
        x1 = (x1 - 127.5) / 128.0
        x2 = (x2 - 127.5) / 128.0
        
        pred = model.predict([x1, x2, x1, x2], verbose=0)
        score = float(pred[0][0])
        
        thresh = st.slider("Ngưỡng", 0.5, 1.0, 0.9)
        st.metric("Độ tương đồng", f"{score:.4f}")
        if score > thresh:
            st.success("CÓ QUAN HỆ HỌ HÀNG")
        else:
            st.warning("KHÔNG CÓ QUAN HỆ")
