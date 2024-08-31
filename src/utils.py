import tempfile
from io import BytesIO
import os
import time
import re
from pytubefix import YouTube
from langdetect import detect_langs
from pydub import AudioSegment
from googletrans import Translator
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_youtube_video(url) -> (BytesIO, dict):
    """
    :param url: address of the YouTube video
    :return: the video as a byte stream and its metadata as a dict
    """
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        video_file = BytesIO()
        stream.stream_to_buffer(video_file)
        video_file.seek(0)
        language = detect_langs(yt.title)
        metadata = {'language_code': language[0].lang,
                    'captions': yt.captions,
                    'description': yt.description}
        return video_file, metadata
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None


def extract_audio(video_stream: BytesIO, output_audio_path="audio.wav"):
    """
    :param video_stream:  the video as a byte stream
    :param output_audio_path: path to save the audio file
    """
    try:
        video_stream.seek(0)
        audio = AudioSegment.from_file(video_stream, format="mp4")
        audio.export(output_audio_path, format="wav")
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def extract_frames(video_stream: BytesIO, frame_rate=1, similarity_threshold=0.95):
    """
    :param video_stream: video as a byte stream.
    :param frame_rate: frame rate.
    :param similarity_threshold: a threshold between -1 and 1 - 1 represents that the two frames are fully identical
    :return: list of frames
    """
    try:
        video_stream.seek(0)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(video_stream.read())
            temp_file.flush()
            cap = cv2.VideoCapture(temp_file.name)
        frames, count, fps = [], 0, cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frame_rate)
        prev_frame = None
        while True:
            success, frame = cap.read()
            if not success:
                break
            if count % interval == 0:
                if prev_frame is None:
                    frames.append(frame)
                    prev_frame = frame
                else:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score, _ = compare_ssim(prev_gray, curr_gray, full=True)
                    if score < similarity_threshold:
                        frames.append(frame)
                        prev_frame = frame
            count += 1

        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None


def _split_text(text, max_chunk_size=500):
    """
    :param text: The text to be split.
    :param max_chunk_size: The maximum number of words to split.
    :return: A list of text chunks.
    """
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    # Append the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

t = Translator()

def translator(text:str, src='auto', dest='en', max_chunk_size=500, delay=1.5) -> str:
    """
    :param text: The text to be translated.
    :param src: source language.
    :param dest: destination language.
    :param max_chunk_size: max number of words to translate, separated by sentence boundaries or max amount
    :param delay: time to wait between each translation
    :return: full translated text.
    """
    translated_text = ""
    chunks = _split_text(text, max_chunk_size=max_chunk_size)
    for chunk in chunks:
        try:
            translation = t.translate(chunk, src=src, dest=dest)
            translated_text += translation.text + " "
            time.sleep(delay)
        except Exception as e:
            print(f"Error translating chunk: {e}")
            time.sleep(delay * 2) 
            continue

    return translated_text.strip()


def concatenate_data(audio_transcription, vision_data, metadata):
    # TODO: add a function that concatenate the transcription, metadata and vision data to one paragraph
    return (f'Audio transcription:\n{audio_transcription}\n'
            f'Vision data:\n{vision_data}\n'
            f'Metadata:\n{metadata}\n')