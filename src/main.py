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

        captions = yt.captions
        description = yt.description
        language = detect_langs(yt.title)
        metadata = {'language_code': language[0].lang,
                    'captions': captions,
                    'description': description}
        return new_file, metadata
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def transcribe_audio(file_path, language_code, model=None) -> (dict, dict):
    """
    Transcribes a .wav audio file into text using a SpeechRecognitionModel model from Hugging Face.
    :param: file_path (str): Path to the .wav audio file.
    :param: language_code (str): Language code (e.g., 'en' for English, 'fr' for French).
    :param: model (SpeechRecognitionModel): hugging face model that was pre-downloaded
    :return: str: The transcribed text.
    """
    full_lang = langcodes.Language.get(language_code).display_name().lower()
    if model is None:
        try:
            model = SpeechRecognitionModel(model_path=f"jonatasgrosman/wav2vec2-large-xlsr-53-{full_lang}",
                                           device=device, force_download=True) # elgeish/imvladikon
            transcriptions = model.transcribe([file_path])
        except Exception as e:
            transcriptions = [None]
    else:
        transcriptions = model.transcribe([file_path])
    os.remove(file_path)
    return transcriptions[0], {language_code: model}


def main():
    model_dict = {}
    while True:
        youtube_url = input("Enter YouTube URL (lower 'x' to exit): \n")
        if youtube_url == 'x':
            break
        # youtube_url = "https://www.youtube.com/watch?v=WdOPb8YvxZg"
        audio_file, metadata = download_youtube_audio(youtube_url)
        if audio_file is None:
            print("Failed to download audio. Exiting.")

        print("Audio downloaded. Transcribing...")
        lang_code = metadata.get('language_code')
        text, tmp_dict = transcribe_audio(audio_file, language_code=lang_code, model=model_dict.get(lang_code, None))
        model_dict.update(tmp_dict)
        if text:
            os.remove(audio_file)
        else:
            print('Failed to transcribe audio')
            return
        en_text = oz_translator(text.get('transcription', ''), metadata.get('language_code'))
        while True:
            question = input("ask a question: \n")
            answer = get_gpt_answer(en_text, question, 'an Academic researcher')
            print(f'{question}:: {answer}')
            if input('To exit, type x: \n').lower() == 'x':
                break


if __name__ == "__main__":
    main()

