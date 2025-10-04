# --- 0. Setup ---
# You'll need to install the required libraries first:
# pip install ultralytics opencv-python numpy huggingface_hub supervision deepface google-generativeai elevenlabs

# --- 1. Load Libraries ---
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from deepface import DeepFace
import torch 
import ssl
import os
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import threading
import json

# --- FIX for SSL: CERTIFICATE_VERIFY_FAILED ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -----------------------------------------------

# --- 2. Configure APIs ---
# IMPORTANT: You must set your API Keys as environment variables.
# In your terminal, run:
# export GEMINI_API_KEY="YOUR_API_KEY"
# export ELEVEN_API_KEY="YOUR_ELEVENLABS_KEY"

# Configure Gemini
try:
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("Gemini API configured successfully with gemini-pro.")
except KeyError:
    print("ERROR: GEMINI_API_KEY environment variable not found.")
    gemini_model = None

# Configure ElevenLabs
try:
    eleven_api_key = os.environ["ELEVEN_API_KEY"]
    eleven_client = ElevenLabs(api_key=eleven_api_key)
    print("ElevenLabs API configured successfully.")
except KeyError:
    print("ERROR: ELEVEN_API_KEY environment variable not found.")
    eleven_client = None
# -----------------------------------------------


# --- 3. Download and Load AI Models ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Downloading YOLOv8 face detection model...")
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
print("Model downloaded.")

print("Loading YOLOv8 model...")
yolo_model = YOLO(model_path)

print("Emotion recognition models will be downloaded on first analysis.")
print("All models loaded.")


# --- 4. Helper Function for Song Recommendation & Narration ---
def speak_song_recommendation(emotion, recommendation_box):
    if not gemini_model:
        recommendation_box[0] = "Error: Gemini API not configured."
        return
    if not eleven_client:
        recommendation_box[0] = "Error: ElevenLabs API not configured."
        return

    # Step 1: Get song data from Gemini
    prompt = f"""
    You are a music recommendation expert. Based on the emotion '{emotion}', recommend one great, well-known song.
    Provide the song title and artist.
    Format your response as a JSON object with the keys "title" and "artist".
    Example: {{"title": "Happy", "artist": "Pharrell Williams"}}
    """
    try:
        print(f"Asking Gemini for a '{emotion}' song recommendation...")
        response = gemini_model.generate_content(prompt)
        
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        song_data = json.loads(clean_response)
        
        title = song_data.get('title', 'Unknown Title')
        artist = song_data.get('artist', 'Unknown Artist')

        recommendation_box[0] = f"{title} - {artist}"
        print(f"Gemini recommended: {recommendation_box[0]}")

    except Exception as e:
        print(f"Error processing Gemini API response: {e}")
        recommendation_box[0] = "API Error."
        return

    # Step 2: Create DJ script and generate audio with ElevenLabs
    script = f"I see you're feeling {emotion}. To match that vibe, here's {title} by {artist}. Enjoy!"
    
    try:
        print("Generating audio with ElevenLabs...")
        
        # Replace "YOUR_VOICE_ID_HERE" with the actual Voice ID for your agent.
        audio = eleven_client.generate(
            text=script, 
            voice="IKne3meq5aSn9XLyUdCD",
            model="eleven_multilingual_v2"
        )
        
        print("Playing audio...")
        play(audio)
    except Exception as e:
        print(f"Error with ElevenLabs API: {e}")
        recommendation_box[0] = "Audio Error."


# --- 5. Main Video Processing Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting webcam feed...")
print("Press 's' to get a song recommendation based on your current emotion.")
print("Press 'q' to quit.")

current_emotion = "neutral"
recommendation_box = [""] # Use a list to pass by reference to the thread

while True:
    success, frame = cap.read()
    if not success:
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
    if recommendation_box[0]:
        cv2.putText(frame, "Recommendation:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, recommendation_box[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('AI Music Recommender', frame)

    # --- 6. User Input ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        recommendation_box[0] = "Getting recommendation..."
        # Run the API calls in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=speak_song_recommendation, args=(current_emotion, recommendation_box))
        thread.start()

cap.release()
cv2.destroyAllWindows()

