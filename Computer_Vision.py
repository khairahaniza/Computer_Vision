import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# -----------------------------
# Step 1: Page configuration
# -----------------------------
st.set_page_config(
    page_title="Image Classification with PyTorch & Streamlit",
    layout="centered"
)

st.title("Simple Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** running on **CPU only**.")

# -----------------------------
# Step 2 & 3: CPU configuration
# -----------------------------
device = torch.device("cpu")
st.info("⚙️ This application runs entirely on CPU.")

# -----------------------------
# Step 4: Load ImageNet labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels

labels = load_imagenet_labels()

# -----------------------------
# Step 4: Load pretrained ResNet18
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()          
    model.to(device)      
    return model

model = load_model()

# -----------------------------
# Step 5: Image preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Step 6: Image upload interface
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Step 7: Convert image to tensor & inference
    # -----------------------------
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # -----------------------------
    # Step 8: Top-5 predictions
    # -----------------------------
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("🔍 Top-5 Predictions")

    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability": float(top5_prob[i])
        })
        st.write(
            f"**{labels[top5_catid[i]]}** — "
            f"Probability: {top5_prob[i].item():.4f}"
        )

    # Table display
    st.subheader("📊 Prediction Table")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Step 9: Bar chart visualization
    # -----------------------------
    st.subheader("📈 Prediction Probability Bar Chart")
    chart_df = df.set_index("Class")
    st.bar_chart(chart_df)

else:
    st.info("👆 Please upload an image to start classification.")


