from utils import download_youtube_video
from language_module import get_gpt_answer
from speech_to_text_module import get_audio_transcription, get_initial_asr_models
from vision_module import get_vision_data


def main():
    model_dict = get_initial_asr_models()
    while True:
        youtube_url = input("Enter YouTube URL (lower 'x' to exit): \n")
        if youtube_url == 'x':
            break
        # youtube_url = "https://www.youtube.com/watch?v=WdOPb8YvxZg"
        video, metadata = download_youtube_video(youtube_url)
        if video is None:
            print("Failed to download. Exiting.")
            continue
        audio_transcription = get_audio_transcription(metadata, model_dict, video)
        vision_data = get_vision_data(video)
        while True:
            question = input("ask a question: \n")
            answer = get_gpt_answer(audio_transcription, question)
            print(f'{question}:: {answer}')
            if input('To exit, type x: \nelse, any key').lower() == 'x':
                break


if __name__ == "__main__":
    main()

