import cv2
import numpy as np
from gtts import gTTS
import moviepy.editor as mp

def generate_video(prompt):
    # Your prompt
    #prompt = """Seek wealth, not money or status. Wealth is having assets that earn while you sleep. Money is how we transfer time and wealth. Status is your place in the social hierarchy."""

    # Convert the prompt to speech using gTTS
    tts = gTTS(text=prompt, lang='en')
    tts.save("prompt.mp3")

    # Load the audio clip
    audio_clip = mp.AudioFileClip("prompt.mp3")
    audio_duration = audio_clip.duration

    # Define video parameters
    width, height = 640, 480
    fps = 24

    # Create a blank video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

    # Add audio frames to the video
    for i in range(int(audio_duration * fps)):
        # Create a blank frame
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Write the blank frame to the video
        video_writer.write(blank_frame)

    # Release the video writer
    video_writer.release()

    # Combine audio and video using moviepy
    video_clip = mp.VideoFileClip("output_video.mp4")
    final_clip = video_clip.set_audio(audio_clip)

    # Export the final video
    final_clip.write_videofile("output_video_with_audio.mp4", codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, fps=24)




