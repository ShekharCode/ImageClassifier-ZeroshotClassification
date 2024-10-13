import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load your trained model (replace this with your model)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # Example using CLIP for zero-shot classification
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Define labels
labels = ["Military Content", "Nudity/Sexuality Content", "Religious Content", "National Motifs", "God/Goddess", "Trademark Content", "Neutral/Non-sensitive Content"]

# Define a function to run the model and classify the image
def classify_image(image):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Logits between the image and text
    probs = logits_per_image.softmax(dim=1)  # Apply softmax to get probabilities
    
    # Get the predicted class
    max_prob_index = torch.argmax(probs, dim=1).item()
    predicted_class = labels[max_prob_index]
    
    # Zip labels and probabilities together
    label_prob_pairs = zip(labels, probs[0])
    
    # Sort by probability in descending order
    sorted_label_prob_pairs = sorted(label_prob_pairs, key=lambda x: x[1], reverse=True)
    
    return predicted_class, sorted_label_prob_pairs

# Streamlit app UI
st.title("Multiclass Image Classification")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Classify the image
    st.write("Classifying...")
    predicted_class, sorted_label_prob_pairs = classify_image(image)
    
    # Display the predicted class
    st.write(f"Predicted Class: **{predicted_class}**")
    
    # Display probabilities for all labels
    st.write("### Probabilities for each class:")
    for label, prob in sorted_label_prob_pairs:
        st.write(f"{label}: {prob.item() * 100:.2f}%")
