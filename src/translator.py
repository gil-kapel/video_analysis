from googletrans import Translator
t = Translator()

def oz_translator(text: str, language_code: str) -> str:
    return t.translate(text[:3181], src=language_code, dest='en').text