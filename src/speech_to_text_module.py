import os
import langcodes
import torch
import whisper
import json
from utils import extract_audio, translator

d = "mps" if torch.backends.mps.is_available() else "cpu"
if d == 'cpu':
    d = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(d)
from huggingsound import SpeechRecognitionModel


with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def get_initial_asr_models():
    whisper_model = config.get('whisper_model', None)
    model = whisper.load_model(whisper_model)
    return {'en': model, 'fa': model} if whisper_model else {}


def transcribe_audio(file_path: str, language_code: str, model: SpeechRecognitionModel=None) -> (dict, dict):
    """
    Transcribes a .wav audio file into text using a SpeechRecognitionModel model from Hugging Face.
    :param file_path: Path to the .wav audio file.
    :param language_code: Language code (e.g., 'en' for English, 'fr' for French).
    :param model: (SpeechRecognitionModel) hugging face model that was pre-downloaded
    :return: str: The transcribed text.
    """
    full_lang = langcodes.Language.get(language_code).display_name().lower()
    if model is None:
        try:
            model = SpeechRecognitionModel(model_path=f"jonatasgrosman/wav2vec2-large-xlsr-53-{full_lang}",
                                           device=device) # elgeish/imvladikon
            transcriptions = model.transcribe([file_path])
        except Exception as e:
            print(f"An error occurred: {e}")
            transcriptions = [None]
        result = transcriptions[0].get('transcription', '')
    else:
        transcriptions = model.transcribe(file_path, verbose=True)
        result = transcriptions['text']
    os.remove(file_path)
    return result, {language_code: model}


def get_audio_transcription(metadata, model_dict, video):
    audio_path = extract_audio(video)
    print("Audio downloaded. Transcribing...")
    lang_code = metadata.get('language_code')
    text, tmp_dict = transcribe_audio(audio_path, language_code=lang_code, model=model_dict.get(lang_code, None))
    model_dict.update(tmp_dict)
    en_text = translator(text=text, src=metadata.get('language_code'))
    return en_text