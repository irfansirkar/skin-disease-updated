import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os
import json # For parsing JSON response from LLM
import requests # Import the requests library for synchronous API calls

# --- Configuration ---
# Path to your trained YOLOv11 model weights (e.g., best.pt)
# Make sure this file is in the same directory as your Streamlit app,
# or provide the full path to it.
MODEL_PATH = 'best.pt'

# Pre-defined simple explanations for each disease
# These will be displayed initially. For more detailed info, the LLM will be used.
DISEASE_EXPLANATIONS = {
    'Acne': "Acne is a common skin condition that occurs when hair follicles plug with oil and dead skin cells. It often causes pimples, blackheads, or whiteheads.",
    'Chickenpox': "Chickenpox is a highly contagious disease caused by the varicella-zoster virus. It results in an itchy, blister-like rash all over the body.",
    'Eczema': "Eczema (atopic dermatitis) is a condition that causes dry, itchy, and inflamed skin. It's common in children but can occur at any age.",
    'Monkeypox': "Monkeypox is a viral disease with symptoms similar to smallpox, though less severe. It causes a rash that can be mistaken for chickenpox or syphilis.",
    'Pimple': "A pimple is a type of acne lesion, a small pustule or papule. Pimples can develop on the face, back, chest, and shoulders.",
    'Psoriasis': "Psoriasis is a chronic autoimmune disease that causes skin cells to build up rapidly on the surface of the skin, forming thick, silvery scales and itchy, dry, red patches.",
    'Ringworm': "Ringworm is a common fungal infection of the skin that causes a circular rash with clearer skin in the middle. It's highly contagious but not serious.",
    'basal cell carcinoma': "Basal cell carcinoma is a type of skin cancer that begins in the basal cells — a type of cell within the skin that produces new skin cells. It often appears as a pearly or waxy bump on sun-exposed areas.",
    'melanoma': "Melanoma is the most serious type of skin cancer, developing in the cells (melanocytes) that produce melanin. It can develop anywhere on the body and often looks like a mole.",
    'tinea-versicolor': "Tinea versicolor is a common fungal infection that causes small, discolored patches of skin. These patches may be lighter or darker than the surrounding skin.",
    'vitiligo': "Vitiligo is a long-term condition where pale, white patches develop on the skin. It's caused by the lack of melanin, the pigment in skin.",
    'warts': "Warts are small, rough growths on the skin caused by human papillomavirus (HPV) infection. They can appear anywhere on the body."
}

# --- Load the YOLOv11 Model ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_model():
    """Loads the YOLOv11 model from the specified path."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at '{MODEL_PATH}'. "
                 "Please ensure 'best.pt' is in the same directory as this script "
                 "or update the MODEL_PATH variable.")
        return None
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}. Error: {e}")
        return None

# --- Gemini API Call for Detailed Explanation ---
def get_detailed_explanation(disease_name):
    """
    Calls the Gemini API to get a more detailed explanation of the disease.
    """
    prompt = f"Provide a detailed, easy-to-understand explanation of '{disease_name}' skin disease, including its common symptoms, causes, and general advice (not medical advice). Keep it concise, around 150-200 words."
    
    chatHistory = []
    chatHistory.append({"role": "user", "parts": [{"text": prompt}]}) # Use .append() for Python lists
    payload = {"contents": chatHistory}
    apiKey = "AIzaSyCZj3kkYtJO8pTzku2MOBqDwlYCMnONXGM"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.warning("Could not generate detailed explanation. Unexpected API response structure.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding API response: {e}. Response: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("⚕️ Skin Disease Detector")
st.markdown("Upload an image of a skin condition to get a prediction and explanation.")

# Load the model once
model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting...")

        # Convert PIL Image to NumPy array for YOLO
        img_np = np.array(image)

        # Perform inference
        try:
            results = model(img_np, verbose=False) # verbose=False to suppress extensive output
            
            # Assuming 'results' is a list of Results objects (one per image)
            if results and len(results) > 0:
                result = results[0] # Get the first (and only) result object
                
                # Check if any detections were made
                if result.boxes and len(result.boxes) > 0:
                    # Get the most confident detection
                    best_detection = result.boxes[0]
                    class_id = int(best_detection.cls.cpu().numpy())
                    confidence = best_detection.conf.cpu().numpy()
                    
                    # Get class name from model's names attribute
                    predicted_class_name = model.names[class_id]
                    
                    st.success(f"Prediction: **{predicted_class_name}** (Confidence: {confidence[0]:.2f})")
                    
                    # Display simple explanation
                    if predicted_class_name in DISEASE_EXPLANATIONS:
                        st.subheader("What is it?")
                        st.info(DISEASE_EXPLANATIONS[predicted_class_name])
                    else:
                        st.info("No specific explanation found for this disease in our database.")

                    # Option to get detailed explanation from LLM
                    if st.button(f"Tell me more about {predicted_class_name}"):
                        with st.spinner(f"Generating detailed explanation for {predicted_class_name}..."):
                            # Call the synchronous function
                            detailed_info = get_detailed_explanation(predicted_class_name)
                            if detailed_info:
                                st.subheader(f"Detailed Information about {predicted_class_name}")
                                st.write(detailed_info)
                            else:
                                st.warning("Failed to retrieve detailed information.")
                else:
                    st.warning("No skin disease detected in the image.")
            else:
                st.warning("No results returned from the model.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please ensure your model is correctly trained and the image format is compatible.")

else:
    st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure `best.pt` exists.")

st.markdown("---")
st.markdown("Disclaimer: This application is for informational purposes only and should not be used for self-diagnosis or medical advice. Consult a healthcare professional for any health concerns.")
