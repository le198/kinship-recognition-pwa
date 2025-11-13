import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ====================== FORCE CPU ======================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import math
import builtins
builtins.tf = tf
builtins.tensorflow = tf
keras = tf.keras
keras.config.enable_unsafe_deserialization()

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

# Patch functions
def patch_square(x): return tf.math.square(x)
def patch_signed_sqrt(x): return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x) + 1e-9)
def patch_scaling_02(x): return x * 0.2
def patch_scaling_10(x): return x * 1.0

# ====================== PWA SUPPORT ======================
st.set_page_config(
    page_title="Kinship AI",
    page_icon="family",
    layout="centered"
)
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#4CAF50">
""", unsafe_allow_html=True)

# ====================== PREPROCESSING ======================
def prewhiten(x):
    if x.ndim == 4: axis = (1,2,3); size = x[0].size
    elif x.ndim == 3: axis = (0,1,2); size = x.size
    else: raise ValueError('Dim 3 or 4')
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    return (x - mean) / std_adj

def preprocess_input_vggface(img):
    img = img.astype(np.float32)
    img[..., 0] -= 93.5940
    img[..., 1] -= 104.7624
    img[..., 2] -= 129.1863
    return img

def read_img_fn(img): return prewhiten(cv2.resize(img, (160, 160)).astype(np.float32))
def read_img_vgg(img): return preprocess_input_vggface(cv2.resize(img, (224, 224)))

# ====================== FACE DETECTION ======================
FACE_DETECTOR_PATH = "face_detector"
PROTOTXT = os.path.join(FACE_DETECTOR_PATH, "deploy.prototxt")
MODEL_WEIGHTS = os.path.join(FACE_DETECTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel")

def download_file(url, path):
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            st.error(f"Download failed: {e}")

@st.cache_resource
def load_face_detector():
    os.makedirs(FACE_DETECTOR_PATH, exist_ok=True)
    download_file("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", PROTOTXT)
    download_file("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", MODEL_WEIGHTS)
    if not all(os.path.exists(p) for p in [PROTOTXT, MODEL_WEIGHTS]):
        st.error("Face detector files missing.")
        return None
    try:
        return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_WEIGHTS)
    except Exception as e:
        st.error(f"Failed to load face detector: {e}")
        return None

face_net = load_face_detector()

# ====================== CROP & DRAW BBOX ======================
def detect_face_and_draw(img_bgr, conf_threshold=0.5):
    if face_net is None: return img_bgr, None, 0
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    best_box, best_conf = None, 0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box, best_conf = box.astype(int), conf
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{best_conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_bgr, best_box, best_conf

def crop_224_from_center(img_bgr, target_size=224):
    _, best_box, _ = detect_face_and_draw(img_bgr.copy())
    if best_box is None: return None
    x1, y1, x2, y2 = best_box
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0: return None
    return cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

# ====================== LOAD MODEL FROM RELEASE ======================
@st.cache_resource
def load_kinship_model():
    model_path = "facenet_vgg.keras"
    
    if not os.path.exists(model_path):
        release_url = "https://github.com/le198/kinship-recognition-pwa/releases/download/v1.0/facenet_vgg.keras"
        with st.spinner("Đang tải model từ GitHub Release... (lần đầu ~1-2 phút)"):
            try:
                urllib.request.urlretrieve(release_url, model_path)
                st.success("Model tải thành công!")
            except Exception as e:
                st.error(f"Tải model thất bại: {e}")
                return None

    try:
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

        # Patch Lambda layers
        for layer in model.layers:
            if layer.__class__.__name__ == 'Lambda' and hasattr(layer, 'function'):
                name = layer.name.lower()
                if any(k in name for k in ['square', 'lambda_23', 'lambda_24', 'lambda_21', 'lambda_22']):
                    layer.function = patch_square
                elif any(k in name for k in ['signed_sqrt', 'lambda_26', 'lambda_25']):
                    layer.function = patch_signed_sqrt
                elif any(k in name for k in ['scaling', 'lambda_18', 'lambda_19']):
                    scale = 0.2
                    if hasattr(layer, 'arguments') and 'scale' in layer.arguments:
                        scale = layer.arguments['scale']
                    layer.function = lambda x, s=scale: x * s
                elif 'lambda_20' in name:
                    layer.function = patch_scaling_10

        st.success("Model đã sẵn sàng!")
        return model
    except Exception as e:
        st.error("Lỗi load model:")
        import traceback
        st.code(traceback.format_exc())
        return None

model = load_kinship_model()

# ====================== UI & XỬ LÝ ======================
st.title("Kinship Recognition AI")
st.markdown("### Chụp hoặc upload 2 ảnh để kiểm tra quan hệ họ hàng")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Ảnh 1")
    source1 = st.radio("Nguồn", ["Upload", "Webcam"], key="src1", horizontal=True)
    if source1 == "Upload":
        uploaded1 = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"], key="up1")
        img1_input = uploaded1
    else:
        uploaded1 = None
        img1_input = st.camera_input("Chụp ảnh 1", key="cam1")
    resize1 = st.checkbox("Resize (bỏ qua crop)", key="r1")

with col2:
    st.subheader("Ảnh 2")
    source2 = st.radio("Nguồn", ["Upload", "Webcam"], key="src2", horizontal=True)
    if source2 == "Upload":
        uploaded2 = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"], key="up2")
        img2_input = uploaded2
    else:
        uploaded2 = None
        img2_input = st.camera_input("Chụp ảnh 2", key="cam2")
    resize2 = st.checkbox("Resize (bỏ qua crop)", key="r2")

if (uploaded1 or img1_input) and (uploaded2 or img2_input) and model and face_net:
    try:
        img1_bytes = uploaded1.getvalue() if uploaded1 else img1_input.getvalue()
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2_bytes = uploaded2.getvalue() if uploaded2 else img2_input.getvalue()
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Đọc ảnh lỗi: {e}")
        st.stop()

    if img1 is None or img2 is None:
        st.error("Không đọc được ảnh.")
        st.stop()

    with st.spinner("Phát hiện khuôn mặt..."):
        img1_box, box1, conf1 = detect_face_and_draw(img1.copy())
        img2_box, box2, conf2 = detect_face_and_draw(img2.copy())
        face1 = cv2.resize(img1, (224, 224)) if resize1 else crop_224_from_center(img1)
        face2 = cv2.resize(img2, (224, 224)) if resize2 else crop_224_from_center(img2)

    if face1 is None or face2 is None:
        st.error("Không phát hiện khuôn mặt. Vui lòng chụp rõ mặt.")
        st.stop()

    st.subheader("Ảnh Gốc + BBox")
    c1, c2 = st.columns(2)
    with c1: st.image(cv2.cvtColor(img1_box, cv2.COLOR_BGR2RGB), caption=f"Ảnh 1 (conf: {conf1:.2f})", use_container_width=True)
    with c2: st.image(cv2.cvtColor(img2_box, cv2.COLOR_BGR2RGB), caption=f"Ảnh 2 (conf: {conf2:.2f})", use_container_width=True)

    st.subheader("Ảnh Crop (224×224)")
    c3, c4 = st.columns(2)
    with c3: st.image(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB), caption="Face 1", use_container_width=True)
    with c4: st.image(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB), caption="Face 2", use_container_width=True)

    X1_FN = np.expand_dims(read_img_fn(face1), axis=0)
    X2_FN = np.expand_dims(read_img_fn(face2), axis=0)
    X1_VGG = np.expand_dims(read_img_vgg(face1), axis=0)
    X2_VGG = np.expand_dims(read_img_vgg(face2), axis=0)

    try:
        with st.spinner("Tính độ tương đồng..."):
            pred1 = model.predict([X1_FN, X2_FN, X1_VGG, X2_VGG], verbose=0)
            pred2 = model.predict([X2_FN, X1_FN, X2_VGG, X1_VGG], verbose=0)
            score = float((pred1[0][0] + pred2[0][0]) / 2)

        st.markdown("---")
        st.subheader("Kết Quả")
        threshold = st.slider("Ngưỡng", 0.5, 1.0, 0.95, 0.01)
        st.metric("Similarity Score", f"{score:.4f}")

        if score > threshold:
            st.success(f"**CÓ QUAN HỆ HỌ HÀNG** (score > {threshold})")
        else:
            st.warning(f"**KHÔNG CÓ QUAN HỆ** (score ≤ {threshold})")

    except Exception as e:
        st.error(f"Dự đoán lỗi: {e}")

else:
    st.info("Vui lòng chọn 2 ảnh.")
    if not model: st.error("Model chưa tải.")
    if not face_net: st.error("Face detector lỗi.")

st.caption("**PWA**: Mở trên Chrome Android → Add to Home screen → Cài như app!")
