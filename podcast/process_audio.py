from moviepy import *
import os

def split_audio(mp3_file_folder, mp3_chunk_folder):
    # Ensure the chunk folder exists
    os.makedirs(mp3_chunk_folder, exist_ok=True)

    # Process each audio file in the folder
    for file_name in os.listdir(mp3_file_folder):
        if file_name.endswith(".mp3"):
            print(f"Splitting Episode ID: {file_name}")

            # Load the audio file using MoviePy
            audio_path = os.path.join(mp3_file_folder, file_name)
            audio_clip = AudioFileClip(audio_path)

            # Define chunk duration 
            chunk_duration = 30  # seconds
            duration = int(audio_clip.duration)
            start_time = 0

            # Split audio into chunks
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                chunk = audio_clip.subclipped(start_time, end_time)
                
                # Save the chunk
                chunk_file_name = f"{file_name[:-4]}_chunk_{start_time}-{end_time}.mp3"
                chunk_file_path = os.path.join(mp3_chunk_folder, chunk_file_name)
                chunk.write_audiofile(chunk_file_path, codec='mp3')

                # Update start time
                start_time += chunk_duration

            
            audio_clip.close()