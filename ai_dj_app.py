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
from elevenlabs.play import play # Corrected import statement
import threading

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
    # Use the 'gemini-pro-latest' model which is more stable for some API versions
    gemini_model = genai.GenerativeModel('gemini-pro-latest')
    print("Gemini API configured successfully with gemini-pro-latest.")
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
def get_and_speak_recommendation(emotion, recommendation_data):
    if not gemini_model or not eleven_client:
        recommendation_data[0] = "API Not Configured"
        recommendation_data[1] = "Please check your API keys."
        recommendation_data[2] = ""
        return

    # Step 1: Get song data from Gemini
    prompt = f"""
    You are a music recommendation expert. The user is currently feeling '{emotion}'.
    Please provide SHORT responses (max 2-3 words for reasoning, 1-2 words for why).
    Format your response EXACTLY like this:
    REASONING: [Very brief emotion validation - 2-3 words max]
    SONG: [Song Title - Artist]
    WHY: [Why this song fits - 1-2 words max]
    """
    try:
        print(f"Asking Gemini for a '{emotion}' song recommendation...")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"Gemini response: {response_text}")

        # Parse the response and update the UI data
        lines = response_text.split('\n')
        # Set default values
        recommendation_data[0], recommendation_data[1], recommendation_data[2] = "Parsing Error", "Check console", ""
        for line in lines:
            if line.startswith('SONG:'):
                recommendation_data[0] = line.replace('SONG:', '').strip()
            elif line.startswith('REASONING:'):
                recommendation_data[1] = line.replace('REASONING:', '').strip()
            elif line.startswith('WHY:'):
                recommendation_data[2] = line.replace('WHY:', '').strip()

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        recommendation_data[0], recommendation_data[1], recommendation_data[2] = "API Error.", "Check console.", ""
        return

    # Step 2: Create DJ script and generate audio with ElevenLabs
    song, reasoning, why = recommendation_data
    script = f"You seem to be feeling {emotion}. {reasoning}. Here's a recommendation: {song}. It's a great choice because it's {why}."
    
    try:
        print("Generating audio with ElevenLabs...")
        # Using the .text_to_speech.convert() method for older library compatibility
        audio = eleven_client.text_to_speech.convert(
            text=script, 
            voice_id="IKne3meq5aSn9XLyUdCD", # The voice ID from your first script
            model_id="eleven_multilingual_v2"
        )
        print("Playing audio...")
        play(audio)
    except Exception as e:
        print(f"Error with ElevenLabs API: {e}")
        recommendation_data[0] = "Audio Error."
        recommendation_data[1] = "Check console."
        recommendation_data[2] = ""

# --- 5. Main Video Processing Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting webcam feed...")
print("Press 's' to get a song recommendation based on your current emotion.")
print("Press 'q' to quit.")

current_emotion = "neutral"
# Use a mutable list to hold recommendation data for threading
recommendation_data = ["", "", ""] # [song, reasoning, why]

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
        if face.size == 0: continue
            
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            current_emotion = analysis[0]['dominant_emotion']
            status_text = f"Emotion: {current_emotion.capitalize()}"
            status_color = (255, 255, 0) # Cyan
        except:
            status_text = "Analyzing..."
            status_color = (0, 0, 255) # Red

        cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # --- Step B: Display the recommendation UI ---
    if recommendation_data[0]:
        interface_width, interface_height = 500, 140
        start_x = frame.shape[1] - interface_width - 10
        start_y = 10
        
        cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), (0, 255, 255), 2)
        
        cv2.putText(frame, recommendation_data[1], (start_x + 10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, recommendation_data[0], (start_x + 10, start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, recommendation_data[2], (start_x + 10, start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    cv2.imshow('AI DJ App', frame)

    # --- Step C: User Input ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        recommendation_data[:] = ["Getting recommendation...", "Analyzing emotions...", "Please wait..."]
        # Run API calls in a thread to prevent GUI freezing
        thread = threading.Thread(target=get_and_speak_recommendation, args=(current_emotion, recommendation_data))
        thread.start()

cap.release()
cv2.destroyAllWindows()

