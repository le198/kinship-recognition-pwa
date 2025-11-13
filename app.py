import urllib.request
import os

@st.cache_resource
def load_kinship_model():
    model_path = "facenet_vgg.keras"
    
    # Nếu chưa có file → TẢI TỪ GITHUB RELEASE
    if not os.path.exists(model_path):
        release_url = "https://github.com/le198/kinship-recognition-pwa/releases/download/v1.0/facenet_vgg.keras"
        with st.spinner(f"Đang tải model (~{os.path.getsize(model_path)/1e6:.1f}MB nếu có cache)..."):
            try:
                urllib.request.urlretrieve(release_url, model_path)
                st.success("Model tải thành công!")
            except Exception as e:
                st.error(f"Không tải được model: {e}")
                return None

    # Load model như cũ
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

        # PATCH LAMBDA LAYERS
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

        return model
    except Exception as e:
        st.error("Lỗi load model:")
        import traceback
        st.code(traceback.format_exc())
        return None
