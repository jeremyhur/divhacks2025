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
import urllib.request # <-- Added for image downloading

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
    cache_path = ".dj_cache_file"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        cache_path=cache_path
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
        user_id = sp.current_user()['id']
        if DIVHACKS_PLAYLIST_ID is None:
            playlists = sp.current_user_playlists()
            for playlist in playlists['items']:
                if playlist['name'] == "DivHacks":
                    DIVHACKS_PLAYLIST_ID = playlist['id']
                    print("Found existing 'DivHacks' playlist.")
                    break
            if DIVHACKS_PLAYLIST_ID is None:
                print("Creating new 'DivHacks' playlist...")
                playlist = sp.user_playlist_create(user=user_id, name="DivHacks", public=True, description="MoodSwing DJ's recommended songs! Enjoy :)")
                DIVHACKS_PLAYLIST_ID = playlist['id']
                print(f"Playlist 'DivHacks' created with ID: {DIVHACKS_PLAYLIST_ID}")

        if DIVHACKS_PLAYLIST_ID:
            sp.playlist_add_items(DIVHACKS_PLAYLIST_ID, [track_uri])
            print(f"Added track to 'DivHacks' playlist.")

    except Exception as e:
        print(f"An error occurred while managing the Spotify playlist: {e}")

# --- SPOTIFY --- Helper function to play a song and return its URI and Album Art
def play_song_on_spotify(song_title, artist_name):
    if not sp:
        print("Spotify not configured. Cannot play song.")
        return None, None

    try:
        query = f"track:{song_title} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        tracks = results['tracks']['items']
        if not tracks:
            print(f"Could not find '{song_title}' by '{artist_name}' on Spotify.")
            return None, None

        track = tracks[0]
        track_uri = track['uri']
        
        # --- Get the album cover URL ---
        image_url = None
        if track['album']['images']:
            image_url = track['album']['images'][0]['url']
        # --------------------------------

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
            
        return track_uri, image_url

    except Exception as e:
        print(f"An error occurred with Spotify: {e}")
        return None, None

# --- NEW HELPER FUNCTION TO DOWNLOAD AND RESIZE IMAGE ---
def get_image_from_url(url, size=(100, 100)):
    """Downloads an image from a URL and returns it as a resized OpenCV image."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, size)
        return resized_image
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

# --- 4. Helper Function for Song Recommendation & Narration ---
def get_and_speak_recommendation(emotion, recommendation_data):
    if not gemini_model or not eleven_client:
        recommendation_data[0] = "API Not Configured"
        recommendation_data[1] = "Please check your API keys."
        recommendation_data[2] = ""
        return

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

    song, reasoning, why = recommendation_data[0], recommendation_data[1], recommendation_data[2]
    
    # Get album cover immediately after getting the song recommendation (but don't play yet)
    song_title, artist_name = "",""
    if ' - ' in song:
        parts = song.rsplit(' - ', 1)
        song_title = parts[0].strip()
        artist_name = parts[1].strip()

    # Get album cover immediately (but don't start playing the song yet)
    if song_title and artist_name:
        # Search for the song to get album cover
        try:
            query = f"track:{song_title} artist:{artist_name}"
            results = sp.search(q=query, type='track', limit=1)
            
            tracks = results['tracks']['items']
            if tracks:
                track = tracks[0]
                # Get the album cover URL
                image_url = None
                if track['album']['images']:
                    image_url = track['album']['images'][0]['url']
                
                if image_url:
                    album_art = get_image_from_url(image_url, size=(100, 100))
                    recommendation_data[3] = album_art
                    print("Album cover loaded and displayed!")
        except Exception as e:
            print(f"Error getting album cover: {e}")

    # Now generate and play the DJ narration
    script = f"You seem to be feeling {emotion}. {reasoning}. Here's a recommendation: {song}. It's a great choice because it's {why}."
    
    try:
        print("Generating audio with ElevenLabs...")
        audio = eleven_client.text_to_speech.convert(
            text=script, 
            voice_id="IKne3meq5aSn9XLyUdCD",
            model_id="eleven_multilingual_v2"
        )
        print("Playing audio...")
        play(audio)
        
        # NOW play the song after DJ finishes talking
        if song_title and artist_name:
            track_uri, image_url = play_song_on_spotify(song_title, artist_name)
            if track_uri:
                add_song_to_divhacks_playlist(track_uri)

    except Exception as e:
        print(f"Error with ElevenLabs API: {e}")
        recommendation_data[0] = "Audio Error."
        recommendation_data[1] = "Check console."
        recommendation_data[2] = ""

# --- 5. Home Screen ---
def show_home_screen():
    home_width, home_height = 800, 600
    home_frame = np.zeros((home_height, home_width, 3), dtype=np.uint8)
    
    spotify_dark = (25, 20, 20)
    spotify_green = (29, 185, 84)
    spotify_light_green = (30, 215, 96)
    spotify_text = (255, 255, 255)
    spotify_gray = (179, 179, 179)
    
    home_frame[:] = spotify_dark
    
    # --- Load and display logo image ---
    logo_path = "moodswinglogo-removebg-preview.png"  # or "assets/logo.png" if it’s in a subfolder
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is not None:
            # Resize logo to fit the frame nicely
            logo = cv2.resize(logo, (300, 250))  # adjust size as needed

            # Choose where to display (top-left corner here)
            x_offset, y_offset = 240, 5
            y1, y2 = y_offset, y_offset + logo.shape[0]
            x1, x2 = x_offset, x_offset + logo.shape[1]

            # Handle transparency if logo has alpha channel
            if logo.shape[2] == 4:
                alpha_logo = logo[:, :, 3] / 255.0
                alpha_background = 1.0 - alpha_logo
                for c in range(0, 3):
                    home_frame[y1:y2, x1:x2, c] = (
                        alpha_logo * logo[:, :, c] +
                        alpha_background * home_frame[y1:y2, x1:x2, c]
                    )
            else:
                home_frame[y1:y2, x1:x2] = logo
        else:
            print("Warning: Could not load logo image.")
    else:
        print("Warning: logo.png not found.")



    # Define text parameters
    spotify_green = (29, 185, 84)
    spotify_text = (255, 255, 255)
    spotify_gray = (150, 150, 150)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Title Text ---
    title = "Emotion-Based Personalized DJ"
    (text_w, text_h), _ = cv2.getTextSize(title, font, 0.8, 2)
    cv2.putText(home_frame, title, ((home_width - text_w)//2, 240), font, 0.8, spotify_text, 2)

    # --- Divider Line ---
    line_y = 260
    line_width = 300
    cv2.rectangle(
        home_frame,
        ((home_width - line_width)//2, line_y),
        ((home_width + line_width)//2, line_y + 2),
        spotify_green,
        -1
    )

    # --- "Press 'S' to Start" ---
    start_text = "Press 'S' to Start"
    (text_w, text_h), _ = cv2.getTextSize(start_text, font, 1.0, 2)
    cv2.putText(home_frame, start_text, ((home_width - text_w)//2, 300), font, 1.0, spotify_text, 2)

    # --- "Press 'Q' to Quit" ---
    quit_text = "Press 'Q' to Quit"
    (text_w, text_h), _ = cv2.getTextSize(quit_text, font, 1.0, 2)
    cv2.putText(home_frame, quit_text, ((home_width - text_w)//2, 350), font, 1.0, spotify_gray, 2)

    
    cv2.putText(home_frame, "Features:", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spotify_light_green, 2)
    cv2.putText(home_frame, "- Real-time emotion detection", (70, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- AI-powered music recommendations", (70, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- Spotify playback and playlist integration", (70, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    cv2.putText(home_frame, "- ElevenLabs DJ Narration", (70, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_text, 2)
    
    cv2.circle(home_frame, (home_width - 100, 100), 30, spotify_green, -1)
    cv2.circle(home_frame, (home_width - 100, 100), 30, spotify_dark, 3)
    
    play_center_x, play_center_y = home_width - 100, 100
    play_points = np.array([[play_center_x - 8, play_center_y - 12], [play_center_x - 8, play_center_y + 12], [play_center_x + 12, play_center_y]], np.int32)
    cv2.fillPoly(home_frame, [play_points], spotify_dark)
    
    return home_frame

def start_camera_mode():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print("\nStarting webcam feed...")
    print("Press 's' to get a song recommendation based on your current emotion.")
    print("Press 'q' to return to home screen.")

    global current_emotion
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        output = yolo_model(frame, verbose=False)
        detections = Detections.from_ultralytics(output[0])

        face_detected = False
        for box in detections.xyxy:
            face_detected = True
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
            except Exception:
                status_text = "Analyzing..."

            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not face_detected:
            current_emotion = "neutral"

        instructions_y = frame.shape[0] - 80
        cv2.rectangle(frame, (10, instructions_y - 40), (500, instructions_y + 50), (25, 20, 20), -1)
        cv2.rectangle(frame, (10, instructions_y - 40), (500, instructions_y + 50), (29, 185, 84), 2)
        cv2.putText(frame, "Press 'S' for song recommendation", (20, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to return to home", (20, instructions_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (179, 179, 179), 2)

        if recommendation_data[0]:
            interface_width, interface_height = 700, 200
            start_x = 10
            start_y = 10
            
            spotify_dark = (25, 20, 20)
            spotify_green = (29, 185, 84)
            spotify_light_green = (30, 215, 96)
            spotify_text = (255, 255, 255)
            spotify_gray = (179, 179, 179)
            
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_dark, -1)
            cv2.rectangle(frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_green, 3)
            
            # --- Draw the Album Art ---
            album_art_image = recommendation_data[3]
            art_x, art_y = start_x + 20, start_y + 40
            if album_art_image is not None:
                try:
                    art_h, art_w, _ = album_art_image.shape
                    frame[art_y:art_y + art_h, art_x:art_x + art_w] = album_art_image
                except Exception as e:
                    print(f"Error drawing album art: {e}") # Handle potential errors
            
            # --- Adjusted text positions ---
            text_start_x = art_x + 120 
            cv2.rectangle(frame, (text_start_x, start_y + 25), (text_start_x + 20, start_y + 27), spotify_green, -1)
            cv2.putText(frame, recommendation_data[1], (text_start_x + 30, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_text, 2)
            cv2.putText(frame, recommendation_data[0], (text_start_x, start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_light_green, 2)
            cv2.putText(frame, recommendation_data[2], (text_start_x, start_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spotify_gray, 2)

        cv2.imshow('MoodSwing DJ', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Add glowing border effect for Q key with slower fade
            for i in range(5):
                # Create glowing effect with multiple border layers
                glow_frame = frame.copy()
                cv2.rectangle(glow_frame, (5, 5), (glow_frame.shape[1] - 5, glow_frame.shape[0] - 5), (0, 255, 255), 8)  # Cyan glow
                cv2.rectangle(glow_frame, (8, 8), (glow_frame.shape[1] - 8, glow_frame.shape[0] - 8), (255, 255, 255), 4)  # White inner
                cv2.imshow('MoodSwing DJ', glow_frame)
                cv2.waitKey(100)  # Slower pause for smoother fade
            
            cap.release()
            cv2.destroyAllWindows()
            return True
        if key == ord('s'):
            # Add glowing border effect for S key with slower fade
            for i in range(5):
                # Create glowing effect with multiple border layers
                glow_frame = frame.copy()
                cv2.rectangle(glow_frame, (5, 5), (glow_frame.shape[1] - 5, glow_frame.shape[0] - 5), (0, 255, 0), 8)  # Green glow
                cv2.rectangle(glow_frame, (8, 8), (glow_frame.shape[1] - 8, glow_frame.shape[0] - 8), (255, 255, 255), 4)  # White inner
                cv2.imshow('MoodSwing DJ', glow_frame)
                cv2.waitKey(100)  # Slower pause for smoother fade
            
            recommendation_data[:] = ["Getting recommendation...", "Analyzing emotions...", "Please wait...", None]
            
            loading_frame = frame.copy()
            interface_width, interface_height = 700, 200
            start_x = 10
            start_y = 10
            spotify_dark = (25, 20, 20)
            spotify_green = (29, 185, 84)
            spotify_text = (255, 255, 255)
            
            cv2.rectangle(loading_frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_dark, -1)
            cv2.rectangle(loading_frame, (start_x, start_y), (start_x + interface_width, start_y + interface_height), spotify_green, 3)
            cv2.putText(loading_frame, recommendation_data[1], (start_x + 40, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_text, 2)
            cv2.putText(loading_frame, recommendation_data[0], (start_x + 15, start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spotify_green, 2)
            cv2.putText(loading_frame, recommendation_data[2], (start_x + 15, start_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 179, 179), 2)
            
            cv2.imshow('MoodSwing DJ', loading_frame)
            cv2.waitKey(1)
            
            thread = threading.Thread(target=get_and_speak_recommendation, args=(current_emotion, recommendation_data))
            thread.start()

    cap.release()
    cv2.destroyAllWindows()
    return False

# --- Main Application Loop ---
print("Starting MoodSwing DJ App...")
print("Loading home screen...")

current_emotion = "neutral"
recommendation_data = ["", "", "", None] # [Song, Reason, Why, AlbumArt_Image]

while True:
    home_frame = show_home_screen()
    cv2.imshow('MoodSwing DJ', home_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Add glowing border effect for S key on home screen with slower fade
        for i in range(5):
            # Create glowing effect with multiple border layers
            glow_frame = home_frame.copy()
            cv2.rectangle(glow_frame, (5, 5), (glow_frame.shape[1] - 5, glow_frame.shape[0] - 5), (0, 255, 0), 8)  # Green glow
            cv2.rectangle(glow_frame, (8, 8), (glow_frame.shape[1] - 8, glow_frame.shape[0] - 8), (255, 255, 255), 4)  # White inner
            cv2.imshow('MoodSwing DJ', glow_frame)
            cv2.waitKey(100)  # Slower pause for smoother fade
        
        cv2.destroyWindow('MoodSwing DJ')
        if not start_camera_mode():
            break
    elif key == ord('q'):
        # Add glowing border effect for Q key on home screen with slower fade
        for i in range(5):
            # Create glowing effect with multiple border layers
            glow_frame = home_frame.copy()
            cv2.rectangle(glow_frame, (5, 5), (glow_frame.shape[1] - 5, glow_frame.shape[0] - 5), (0, 255, 255), 8)  # Cyan glow
            cv2.rectangle(glow_frame, (8, 8), (glow_frame.shape[1] - 8, glow_frame.shape[0] - 8), (255, 255, 255), 4)  # White inner
            cv2.imshow('MoodSwing DJ', glow_frame)
            cv2.waitKey(100)  # Slower pause for smoother fade
        
        print("Exiting application...")
        break

cv2.destroyAllWindows()