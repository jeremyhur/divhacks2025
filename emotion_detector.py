# --- 0. Setup ---
# You'll need to install the required libraries first:
# pip install ultralytics opencv-python numpy huggingface_hub supervision deepface google-generativeai

# --- 1. Load Libraries ---
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from deepface import DeepFace
import torch 
import ssl
import os
import google.generativeai as genai # Import Gemini API library

# --- FIX for SSL: CERTIFICATE_VERIFY_FAILED ---
# This block allows model files to be downloaded on systems with SSL certificate issues (common on macOS).
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -----------------------------------------------

# --- 2. Configure Gemini API ---
# IMPORTANT: You must set your API Key as an environment variable.
# In your terminal, run: export GEMINI_API_KEY="YOUR_API_KEY"
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Try to list available models first
    try:
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"Could not list models: {e}")
    
    # Use the 'gemini-pro-latest' model which is available
    gemini_model = genai.GenerativeModel('gemini-pro-latest')
    print("Gemini API configured successfully with gemini-pro-latest.")
except KeyError:
    print("ERROR: GEMINI_API_KEY environment variable not found.")
    print("Please set your API key before running the script.")
    gemini_model = None
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_model = None
# -----------------------------------------------


# --- 3. Download and Load AI Models ---

# Check for GPU and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Download the YOLOv8 face detection model from Hugging Face
print("Downloading YOLOv8 face detection model...")
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
print("Model downloaded.")

# Load the YOLOv8 model
print("Loading YOLOv8 model...")
yolo_model = YOLO(model_path)

# DeepFace will automatically download its required models on the first run.
print("Emotion recognition models will be downloaded on first analysis.")
print("All models loaded.")


# --- 4. Helper Function for Song Recommendation ---
def get_song_recommendation(emotion):
    if not gemini_model:
        return "Gemini API not configured."
        
    prompt = f"""
    You are a music recommendation expert.
    Based on the emotion '{emotion}', recommend one great song from Spotify.
    The song should be well-known and fit the mood perfectly.
    Please format your response ONLY as: Song Title - Artist
    """
    try:
        print(f"Asking Gemini for a '{emotion}' song recommendation...")
        response = gemini_model.generate_content(prompt)
        # Clean up the response text
        song = response.text.strip()
        print(f"Gemini recommended: {song}")
        return song
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "API Error."


# --- 5. Main Video Processing Loop ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting webcam feed...")
print("Press 's' to get a song recommendation based on your current emotion.")
print("Press 'q' to quit.")

current_emotion = "neutral"
song_recommendation = ""

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    # --- Step A: Face Detection with YOLOv8 ---
    output = yolo_model(frame, verbose=False)
    detections = Detections.from_ultralytics(output[0])

    if len(detections.xyxy) == 0:
        current_emotion = "neutral"

    for box in detections.xyxy:
        x1, y1, x2, y2 = map(int, box)
        
        padding = 20
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue
            
        # --- Step B: Emotion Recognition with DeepFace ---
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            current_emotion = analysis[0]['dominant_emotion']
            status_text = f"Emotion: {current_emotion.capitalize()}"
            status_color = (255, 255, 0) # Cyan
        except Exception as e:
            status_text = "Analyzing..."
            status_color = (0, 0, 255) # Red

        cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # Display the song recommendation if it exists
    if song_recommendation:
        cv2.putText(frame, "Recommendation:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, song_recommendation, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('AI Music Recommender', frame)

    # --- 6. User Input ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        song_recommendation = "Getting recommendation..."
        # Update the frame immediately to show the "getting recommendation" message
        cv2.putText(frame, "Recommendation:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, song_recommendation, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('AI Music Recommender', frame)
        cv2.waitKey(1) # Allow UI to refresh
        
        # Get the actual recommendation
        song_recommendation = get_song_recommendation(current_emotion)


cap.release()
cv2.destroyAllWindows()

