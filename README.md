# YouTube Video Analysis and Query System

This repository provides an algorithm to analyze YouTube videos by extracting captions, metadata, speech, and vision components. After analyzing, it allows users to ask questions about the movie using a Large Language Model (LLM).

## Table of Contents

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Usage](#usage)
- [Model Downloads](#model-downloads)
- [Folder Structure](#folder-structure)
- [License](#license)

## Installation

### Requirements

Ensure you have Python 3.9 installed. The required Python packages are listed in the `requirements.txt` file.

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gil-kapel/video_analysis.git
   cd video_analysis
   ```

2. **Navigate to the `src` folder:**

   ```bash
   cd src
   ```

3. **Install the required packages:**

   Use `pip` to install the dependencies:

   ```bash
   pip install -r requirements.txt
   pip install googletrans
   pip install openai==0.27.0
   pip install torch torchvision torchaudio huggingsound       
   ```

   Alternatively, if you're using a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   pip install googletrans
   pip install openai==0.27.0
   pip install torch torchvision torchaudio huggingsound 
   ```

4. **Download the necessary models:**

   This project requires certain models for speech recognition, vision analysis, and LLM querying. Follow the URLs below to download and set up the models:

   - **Speech Recognition Model**: [Download Link](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
   - **Vision Analysis Model**:
   - **LLM for Querying**: 


   After downloading, place the models in the `models/` directory within the `src` folder.

5. **Update config.json file:**
   - Update your OpenAI API key
   - Change the agent character as you wish

## Usage

Once you've set up the environment and downloaded the models, you can run the algorithm to analyze a YouTube video and query it.

1. **Run the analysis:**

   ```bash
   python main.py
   ```

2. **Ask questions:**

   After the analysis is complete, you'll be able to interact with the LLM to ask questions about the movie. Follow the on-screen prompts to enter your queries.


## Folder Structure

```
your-project/
│
├── src/
│   ├── main.py                  
│   ├── utils.py                   
│   ├── speech_to_text_module.py  
│   ├── vision_module.py    
│   ├── config.json
│   └── requirements.txt     
│
├── models/
└── README.md            
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---