#pip install git+https://github.com/openai/whisper.git
import os
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm
import whisper
from main import generate_video

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

import shutil

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {e.strerror}")


class VideoTranscriber:
    def __init__(self, model_path, video_path):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = ''
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def transcribe_video(self):
        print('Transcribing video')
        result = self.model.transcribe(self.audio_path)
        text = result["segments"][0]["text"]
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = 16 / 9
        ret, frame = cap.read()
        width = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)].shape[1]
        width = width - (width * 0.1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))

        for j in tqdm(result["segments"]):
            lines = []
            text = j["text"]
            end = j["end"]
            start = j["start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            i = 0

            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = len(words[i]) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i]

                while remaining_pixels > 0:
                    i += 1
                    if i >= len(words):
                        break
                    length_in_pixels = len(words[i]) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]

                line_array = [line, int(start) + 15, int(len(line) / total_chars * total_frames) + int(start) + 15]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)

        cap.release()
        print('Transcription complete')

    def extract_audio(self, output_audio_path):
        print('Extracting audio')
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path)
        self.audio_path = output_audio_path
        print('Audio extracted')

    def extract_frames(self, output_folder, background_image_path):
        print('Extracting frames')
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = width / height
        N_frames = 0

        background_image = cv2.imread(background_image_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)]

            # Resize background image to match frame dimensions
            background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

            background_image = cv2.resize(background_image, (width, height))


            # Blend frame and background
            blended_frame = cv2.addWeighted(frame, 0.5, background_resized, 0.5, 0)

            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x = int((blended_frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height / 2)
                    cv2.putText(blended_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    break

            cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), blended_frame)
            N_frames += 1

        cap.release()
        print('Frames extracted')


    def create_video(self, output_video_path, background_image_path):
        print('Creating video')
        image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        self.extract_frames(image_folder, background_image_path)

        print("Video saved at:", output_video_path)
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(x.split(".")[0]))

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        clip = ImageSequenceClip([os.path.join(image_folder, image) for image in images], fps=self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path)



# Example usage
prompt = """Seek wealth, not money or status. Wealth is having assets that earn while you sleep. Money is how we transfer time and wealth. Status is your place in the social hierarchy."""
generate_video(prompt)

model_path = "base"
video_path = "output_video_with_audio.mp4"
output_video_path = "prompt_video.mp4"
output_audio_path = "prompt_audio.mp3"
background_image_path = "download.jpg"
transcriber = VideoTranscriber(model_path, video_path)
transcriber.extract_audio(output_audio_path=output_audio_path)
transcriber.transcribe_video()
transcriber.create_video(output_video_path,background_image_path)

os.remove("prompt_audio.mp3")
os.remove("output_video_with_audio.mp4")
os.remove("prompt.mp3")
os.remove("output_video.mp4")

# Example usage
delete_folder("frames")