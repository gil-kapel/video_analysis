from utils import download_youtube_video, concatenate_data
from language_module import get_gpt_answer
from speech_to_text_module import get_audio_transcription, get_initial_asr_models
from vision_module import get_vision_data

#TODO: Test plan
#TODO: API that saves the results as a dataset sample + user feedback!!!
def main():
    #TODO: CLI improvement
    model_dict = get_initial_asr_models() # Automatic Speech Recognition
    while True:
        youtube_url = input("Enter 'x' to exit \n Enter YouTube URL: \n")
        if youtube_url.replace(" ", "").lower() == 'x':
            break
        video, metadata = download_youtube_video(youtube_url)
        if video is None:
            print("Failed to download. Exiting.")
            continue
        audio_transcription = get_audio_transcription(metadata, model_dict, video)
        vision_data = get_vision_data(video)
        final_paragraph = concatenate_data(audio_transcription, vision_data, metadata)
        while True:
            question = input("Enter 'x' to exit \n Ask a question: \n")
            if question.lower().replace(" ", "") == 'x':
                break
            answer = get_gpt_answer(final_paragraph, question)
            print(f'{question}:: {answer}')

if __name__ == "__main__":
    main()

# youtube_url = "https://www.youtube.com/watch?v=WdOPb8YvxZg", "https://www.youtube.com/shorts/qm2fJvCyjnM"
