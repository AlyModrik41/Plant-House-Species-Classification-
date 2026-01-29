import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="🌿 House Plant Classifier",
    layout="centered"
)
st.title("🌿 House Plant Species Classification")
st.write("Upload an image of a plant and get its species prediction.")

# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="plant_classifier_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------------------------------
# Load class names from txt
# --------------------------------------------------
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.write(f"✅ Loaded {len(class_names)} classes")  # sanity check

# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # --------------------------------------------------
    # Preprocess image
    # --------------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Sanity check
    st.write("Input shape:", input_data.shape, "dtype:", input_data.dtype)

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # --------------------------------------------------
    # Softmax to get probabilities
    # --------------------------------------------------
    probs = tf.nn.softmax(output).numpy()  # values between 0 and 1, sum to 1

    # --------------------------------------------------
    # Top-1 prediction
    # --------------------------------------------------
    top_index = np.argmax(probs)
    top_class = class_names[top_index]
    top_score = probs[top_index]  # raw softmax score (0-1)

    st.success(f"🌿 Prediction: **{top_class}**")
    st.write(f"📊 Confidence (raw score): **{top_score:.2f}**")

    # --------------------------------------------------
    # Top-3 predictions
    # --------------------------------------------------
    st.subheader("Top 3 Predictions")
    top3_idx = np.argsort(probs)[-3:][::-1]
    for idx in top3_idx:
        st.write(f"- {class_names[idx]} — {probs[idx]:.2f}")

    st.markdown("---")
