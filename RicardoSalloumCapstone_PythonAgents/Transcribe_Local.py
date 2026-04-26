import requests
import tkinter as tk
from tkinter import filedialog
import os

# Configuration - Ensure your Agent_Voice.py is running on this port
API_URL = "http://127.0.0.1:5005/transcribe"

def run_transcription():
    # 1. Open file browser to select .mp3
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    
    print("Please select an .mp3 file...")
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    # 2. Prepare the request
    print(f"Sending '{os.path.basename(file_path)}' to API...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/mpeg')}
            # is_snippet=False tells your API it's a full file, not a 30s Unity chunk
            data = {'is_snippet': 'false'} 
            
            response = requests.post(API_URL, files=files, data=data)
            
        # 3. Handle and print the response
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*30)
            print("TRANSCRIPTION SUCCESSFUL")
            print("="*30)
            print(f"Original: {result.get('original')}")
            print(f"Cleaned:  {result.get('cleaned')}")
            print(f"Processing Time: {result.get('processing_time_seconds')}s")
            print("="*30)
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_transcription()