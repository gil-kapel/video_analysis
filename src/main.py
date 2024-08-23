import os
from pytubefix import YouTube
from langdetect import detect_langs
from pydub import AudioSegment
import langcodes
from nlp_model import get_gpt_answer
from translator import oz_translator

import torch
d = "mps" if torch.backends.mps.is_available() else "cpu"
if d == 'cpu':
    d = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(d)
from huggingsound import SpeechRecognitionModel


def download_youtube_audio(url):
    try:
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        if video is None:
            raise Exception("No audio streams available for this video.")
        out_file = video.download(output_path=".")
        base, ext = os.path.splitext(out_file)
        audio = AudioSegment.from_file(out_file)
        os.remove(out_file)

        new_file = base + '.wav'
        audio.export(new_file, format="wav")
        language = detect_langs(yt.title)
        return new_file, language[0].lang
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def transcribe_audio(file_path, language_code):
    """
    Transcribes a .wav audio file into text using a SpeechRecognitionModel model from Hugging Face.
    Args:
        file_path (str): Path to the .wav audio file.
        language_code (str): Language code (e.g., 'en' for English, 'fr' for French).
    Returns:
        str: The transcribed text.
    """
    full_lang = langcodes.Language.get(language_code).display_name().lower()
    print(f"Using device: {device}")
    try:
        model = SpeechRecognitionModel(f"jonatasgrosman/wav2vec2-large-xlsr-53-{full_lang}", device=device)
        transcriptions = model.transcribe([file_path])
    except Exception as e:
        os.remove(file_path)
        return None
    return transcriptions[0]


def main():
    youtube_url = input("Enter YouTube URL: ")
    # youtube_url = "https://www.youtube.com/watch?v=WdOPb8YvxZg"

    audio_file, lang_code = download_youtube_audio(youtube_url)
    if audio_file is None:
        print("Failed to download audio. Exiting.")

    print("Audio downloaded. Transcribing...")
    text = transcribe_audio(audio_file, language_code=lang_code)
    if text:
        os.remove(audio_file)
    else:
        print('Failed to transcribe audio')
        return
    en_text = oz_translator(text['transcription'], lang_code)
    while True:
        question = input("ask a question: \n")
        answer = get_gpt_answer(en_text, question)
        print(f'{question}:: {answer}')
        if input('To exit, type x: ').lower() == 'x':
            break


if __name__ == "__main__":
    main()

