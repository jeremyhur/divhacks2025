# --- 0. Setup ---
# pip install ultralytics opencv-python numpy huggingface_hub supervision deepface google-generativeai elevenlabs spotipy

# --- 1. Load Libraries ---
try:
    import cv2
except ImportError:
    raise SystemExit("ERROR: OpenCV (cv2) is not installed. Install dependencies with:\n/usr/bin/python3 -m pip install --upgrade -r requirements.txt")
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from deepface import DeepFace
import torch 
import ssl
import os
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import threading
import spotipy # --- SPOTIFY --- Import Spotipy
from spotipy.oauth2 import SpotifyOAuth # --- SPOTIFY --- Import for authentication

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
# export SPOTIPY_CLIENT_ID="YOUR_SPOTIFY_CLIENT_ID"
# export SPOTIPY_CLIENT_SECRET="YOUR_SPOTIFY_CLIENT_SECRET"
# export SPOTIPY_REDIRECT_URI="http://localhost:8888/callback"

# Configure Gemini
try:
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro-latest')
    print("Gemini API configured successfully.")
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

# --- SPOTIFY --- Configure Spotipy
try:
    # Set the scope: permissions your script is asking for.
    # 'user-modify-playback-state' is needed to control playback.
    scope = "user-read-playback-state user-modify-playback-state"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    print("Spotify API configured successfully.")
except Exception as e:
    print(f"ERROR: Could not configure Spotify. Check environment variables. Details: {e}")
    sp = None
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

# --- SPOTIFY --- New helper function to play a song
def play_song_on_spotify(song_title, artist_name):
    if not sp:
        print("Spotify not configured. Cannot play song.")
        return

    try:
        # Search for the track
        query = f"track:{song_title} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        tracks = results['tracks']['items']
        if not tracks:
            print(f"Could not find '{song_title}' by '{artist_name}' on Spotify.")
            return

        track_uri = tracks[0]['uri']
        
        # Find an active device to play on
        devices = sp.devices()
        active_device_id = None
        if devices and devices['devices']:
             for device in devices['devices']:
                if device['is_active']:
                    active_device_id = device['id']
                    break
        
        if not active_device_id:
            print("No active Spotify device found. Please open Spotify on a device.")
            # Optional: Start playback on the first available device if none are active
            # if devices and devices['devices']:
            #    active_device_id = devices['devices'][0]['id']

        if active_device_id:
            print(f"Playing on device ID: {active_device_id}")
            sp.start_playback(device_id=active_device_id, uris=[track_uri])
            print(f"Playing '{song_title}' on Spotify.")
        else:
            print("Could not start playback. No devices available.")

    except Exception as e:
        print(f"An error occurred with Spotify: {e}")


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
    Please provide SHORT responses (max 4-5 words for reasoning, 3-4 words for why).
    Format your response EXACTLY like this:
    REASONING: [Very brief emotion validation - 4-5 words max]
    SONG: [Song Title - Artist]
    WHY: [Why this song fits - 3-4 words max]
    """
    try:
        print(f"Asking Gemini for a '{emotion}' song recommendation...")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"Gemini response: {response_text}")

        lines = response_text.split('\n')
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
    
    # --- SPOTIFY --- Prepare to play the song after DJ intro
    # Parse the 'Song Title - Artist' string from Gemini
    song_title, artist_name = "",""
    if ' - ' in song:
        parts = song.rsplit(' - ', 1)
        song_title = parts[0].strip()
        artist_name = parts[1].strip()

    try:
        print("Generating audio with ElevenLabs...")
        audio = eleven_client.text_to_speech.convert(
            text=script, 
            voice_id="IKne3meq5aSn9XLyUdCD",
            model_id="eleven_multilingual_v2"
        )
        print("Playing audio...")
        play(audio)
        
        # --- SPOTIFY --- Play the song after the narration is finished
        if song_title and artist_name:
            play_song_on_spotify(song_title, artist_name)
        else:
            print("Could not parse song and artist to play on Spotify.")

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
recommendation_data = ["", "", ""]

while True:
    success, frame = cap.read()
    if not success:
        break

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
            status_color = (255, 255, 0)
        except:
            status_text = "Analyzing..."
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

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

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        recommendation_data[:] = ["Getting recommendation...", "Analyzing emotions...", "Please wait..."]
        thread = threading.Thread(target=get_and_speak_recommendation, args=(current_emotion, recommendation_data))
        thread.start()

cap.release()
cv2.destroyAllWindows()