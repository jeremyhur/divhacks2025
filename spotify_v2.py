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
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np

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
DIVHACKS_PLAYLIST_ID = None 
try:
    scope = "user-read-playback-state user-modify-playback-state playlist-modify-public playlist-modify-private"
    
    # --- ADD THIS PARAMETER to force the cache file name ---
    cache_path = ".dj_cache_file" 
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope, 
        cache_path=cache_path # Use the specified cache file
    ))
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

# --- SPOTIFY --- New helper function to manage the "DivHacks" playlist
def add_song_to_divhacks_playlist(track_uri):
    global DIVHACKS_PLAYLIST_ID
    if not sp or not track_uri:
        print("Spotify not configured or no track URI provided. Cannot manage playlist.")
        return

    try:
        # Step 1: Get the current user ID
        user_id = sp.current_user()['id']

        # Step 2: Check/Create the "DivHacks" playlist
        if DIVHACKS_PLAYLIST_ID is None:
            # Check for existing playlist by name
            playlists = sp.current_user_playlists()
            for playlist in playlists['items']:
                if playlist['name'] == "DivHacks":
                    DIVHACKS_PLAYLIST_ID = playlist['id']
                    print("Found existing 'DivHacks' playlist.")
                    break
            
            # If still not found, create a new one
            if DIVHACKS_PLAYLIST_ID is None:
                print("Creating new 'DivHacks' playlist...")
                playlist = sp.user_playlist_create(user=user_id, name="DivHacks", public=True, description="MoodSwing DJ's recommended songs! Enjoy :)")
                DIVHACKS_PLAYLIST_ID = playlist['id']
                print(f"Playlist 'DivHacks' created with ID: {DIVHACKS_PLAYLIST_ID}")

        # Step 3: Add the song to the playlist
        if DIVHACKS_PLAYLIST_ID:
            sp.playlist_add_items(DIVHACKS_PLAYLIST_ID, [track_uri])
            print(f"Added track to 'DivHacks' playlist.")

    except Exception as e:
        print(f"An error occurred while managing the Spotify playlist: {e}")


# --- SPOTIFY --- Helper function to play a song and return its URI
def play_song_on_spotify(song_title, artist_name):
    if not sp:
        print("Spotify not configured. Cannot play song.")
        return None

    try:
        # Search for the track
        query = f"track:{song_title} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        tracks = results['tracks']['items']
        if not tracks:
            print(f"Could not find '{song_title}' by '{artist_name}' on Spotify.")
            return None

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

        if active_device_id:
            print(f"Playing on device ID: {active_device_id}")
            sp.start_playback(device_id=active_device_id, uris=[track_uri])
            print(f"Playing '{song_title}' on Spotify.")
            
        return track_uri # Return the URI for playlist addition

    except Exception as e:
        print(f"An error occurred with Spotify: {e}")
        return None


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
            # The play_song_on_spotify function now returns the track_uri
            track_uri = play_song_on_spotify(song_title, artist_name)
            
            # --- NEW: Add song to the "DivHacks" playlist ---
            if track_uri:
                add_song_to_divhacks_playlist(track_uri)
            # ---------------------------------------------------
        else:
            print("Could not parse song and artist to play on Spotify.")

    except Exception as e:
        print(f"Error with ElevenLabs API: {e}")
        recommendation_data[0] = "Audio Error."
        recommendation_data[1] = "Check console."
        recommendation_data[2] = ""


# --- 5. Home Screen ---
def show_home_screen():
    """Display a beautiful home screen with options"""
    home_width, home_height = 800, 600
    home_frame = np.zeros((home_height, home_width, 3), dtype=np.uint8)
    
    # Spotify colors
    spotify_dark = (25, 20, 20)  # #191414
    spotify_green = (29, 185, 84)  # #1DB954
    spotify_light_green = (30, 215, 96)  # #1ED760
    spotify_text = (255, 255, 255)  # White text
    spotify_gray = (179, 179, 179)  # #B3B3B3
    
    # Background
    home_frame[:] = spotify_dark
    
    # Title
    cv2.putText(home_frame, "MoodSwing", (home_width//2 - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, spotify_light_green, 3)
    cv2.putText(home_frame, "Emotion-Based Personalized DJ", (home_width//2 - 200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spotify_text, 2)
    
    # Decorative line
    cv2.rectangle(home_frame, (home_width//2 - 150, 220), (home_width//2 + 150, 222), spotify_green, -1)
    
    # Options
    cv2.putText(home_frame, "Press 'S' to Start", (home_width//2 - 120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, spotify_text, 2)
    cv2.putText(home_frame, "Press 'Q' to Quit", (home_width//2 - 100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, spotify_gray, 2)
    
    # Features list
    cv2.putText(home_frame, "Features:", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spotify_light_green, 2)
    cv2.putText(home_frame, "- Real-time emotion detection", (70, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- AI-powered music recommendations", (70, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- Spotify playback and playlist integration", (70, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- ElevenLabs DJ Narration", (70, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    
    # Spotify logo area (simplified)
    cv2.circle(home_frame, (home_width - 100, 100), 30, spotify_green, -1)
    cv2.circle(home_frame, (home_width - 100, 100), 30, spotify_dark, 3)
    
    # Play button in logo
    play_center_x, play_center_y = home_width - 100, 100
    play_points = np.array([
        [play_center_x - 8, play_center_y - 12],
        [play_center_x - 8, play_center_y + 12],
        [play_center_x + 12, play_center_y]
    ], np.int32)
    cv2.fillPoly(home_frame, [play_points], spotify_dark)
    
    return home_frame

def start_camera_mode():
    """Start the camera and emotion detection mode"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print("\nStarting webcam feed...")
    print("Press 's' to get a song recommendation based on your current emotion.")
    print("Press 'q' to return to home screen.")

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

            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add instructions overlay
        instructions_y = frame.shape[0] - 80
        cv2.rectangle(frame, (10, instructions_y - 40), (500, instructions_y + 50), (25, 20, 20), -1)  # Dark background
        cv2.rectangle(frame, (10, instructions_y - 40), (500, instructions_y + 50), (29, 185, 84), 2)  # Green border
        
        # Instructions text
        cv2.putText(frame, "Press 'S' for song recommendation", (20, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to return to home", (20, instructions_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (179, 179, 179), 2)

        if recommendation_data[0]:
            # Spotify-themed interface
            interface_width, interface_height = 520, 160
            start_x = frame.shape[1] - interface_width - 10
            start_y = 10
            
            # Spotify colors: Dark background (#191414), Green accent (#1DB954)
            spotify_dark = (25, 20, 20)  # #191414
            spotify_green = (29, 185, 84)  # #1DB954
            spotify_light_green = (30, 215, 96)  # #1ED760
            spotify_text = (255, 255, 255)  # White text
            spotify_gray = (179, 179, 179)  # #B3B3B3
            
            # Main interface background
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_dark, -1)
            
            # Spotify green border
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_green, 3)
            
            # Add Spotify-style rounded corners effect (simplified)
            cv2.circle(frame, (start_x + 5, start_y + 5), 5, spotify_dark, -1)
            cv2.circle(frame, (start_x + interface_width - 5, start_y + 5), 5, spotify_dark, -1)
            cv2.circle(frame, (start_x + 5, start_y + interface_height - 5), 5, spotify_dark, -1)
            cv2.circle(frame, (start_x + interface_width - 5, start_y + interface_height - 5), 5, spotify_dark, -1)
            
            # Spotify green accent line
            cv2.rectangle(frame, (start_x + 10, start_y + 25), (start_x + 30, start_y + 27), spotify_green, -1)
            
            # Emotion text with FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, recommendation_data[1], (start_x + 40, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_text, 2)
            
            # Song title with Spotify green
            cv2.putText(frame, recommendation_data[0], (start_x + 15, start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_light_green, 2)
            
            # Why text in gray
            cv2.putText(frame, recommendation_data[2], (start_x + 15, start_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_gray, 2)
            
            # Add Spotify-style play button (triangle)
            play_center_x = start_x + interface_width - 30
            play_center_y = start_y + 30
            
            # Draw play button background circle
            cv2.circle(frame, (play_center_x, play_center_y), 12, spotify_green, -1)
            cv2.circle(frame, (play_center_x, play_center_y), 12, spotify_dark, 2)
            
            # Draw play triangle inside the circle
            play_points = np.array([
                [play_center_x - 4, play_center_y - 6],  # Left point
                [play_center_x - 4, play_center_y + 6],  # Bottom left
                [play_center_x + 6, play_center_y]       # Right point
            ], np.int32)
            cv2.fillPoly(frame, [play_points], spotify_dark)

        cv2.imshow('AI DJ App - Home', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return True  # Return to home screen
        if key == ord('s'):
            recommendation_data[:] = ["Getting recommendation...", "Analyzing emotions...", "Please wait..."]
            
            # Show loading state with Spotify theme
            interface_width, interface_height = 520, 160
            start_x = frame.shape[1] - interface_width - 10
            start_y = 10
            
            # Spotify colors
            spotify_dark = (25, 20, 20)  # #191414
            spotify_green = (29, 185, 84)  # #1DB954
            spotify_text = (255, 255, 255)  # White text
            
            # Loading interface background
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_dark, -1)
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_green, 3)
            
            # Loading text
            cv2.putText(frame, recommendation_data[1], (start_x + 40, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_text, 2)
            cv2.putText(frame, recommendation_data[0], (start_x + 15, start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_green, 2)
            cv2.putText(frame, recommendation_data[2], (start_x + 15, start_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 179, 179), 2)
            
            cv2.imshow('AI DJ App - Home', frame)
            cv2.waitKey(1)
            
            thread = threading.Thread(target=get_and_speak_recommendation, args=(current_emotion, recommendation_data))
            thread.start()

    cap.release()
    cv2.destroyAllWindows()

# Show home screen
print("Starting AI DJ App...")
print("Loading home screen...")

current_emotion = "neutral"
recommendation_data = ["", "", ""]

# Main application loop
while True:
    # Show home screen
    home_frame = show_home_screen()
    cv2.imshow('AI DJ App - Home', home_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.destroyWindow('AI DJ App - Home')
        if start_camera_mode():
            continue  # Return to home screen
        else:
            break  # Exit if camera failed
    elif key == ord('q'):
        print("Exiting application...")
        break

cv2.destroyAllWindows()