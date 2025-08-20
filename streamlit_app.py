import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# إعداد الصفحة
st.set_page_config(page_title="🌼 Flower Classifier", page_icon="🌻", layout="centered")

# تحميل الموديل واللابلز
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("flowers_model.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # عكس القاموس {class_name: index} -> {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

# معالجة الصورة
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")       # تحويل لـ RGB
    img = img.resize((128, 128))   # تغيير الحجم حسب الموديل
    arr = np.array(img) / 255.0    # نفس rescale اللي في التدريب
    arr = np.expand_dims(arr, axis=0)  # (1,128,128,3)
    return arr

# واجهة Streamlit
st.title("🌼 Flower Classifier")
st.caption("CNN model trained on: daisy, dandelion, rose, sunflower, tulip")

uploaded = st.file_uploader("Input your image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)

    arr = preprocess_image(img)
    preds = model.predict(arr)[0]   # احتمالات كل كلاس

    top_idx = int(np.argmax(preds))
    top_class = idx_to_class[top_idx]
    confidence = preds[top_idx]

    st.subheader(f"🌸Prediction : **{top_class}**")
    st.write(f" Accuracy: **{confidence:.2%}**")

    # رسم البار تشارت
    import pandas as pd
    df = pd.DataFrame({
        "class": [idx_to_class[i] for i in range(len(preds))],
        "probability": preds
    })
    st.bar_chart(df.set_index("class"))
