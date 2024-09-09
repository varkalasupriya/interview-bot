# Introduction  
Recruitment Bot is a project to automate interview processes.
It involves: 
- Parsing a resume: Using pypdf to read the pdf
- LangGraph: Using the llama-v3 open source model on HuggingFace
- Text-to-speech: Using xTTS (can give desired voice of the audio)
- Created as an API: (using FastAPI)
- Currently streamlit is the frontend 

# Getting Started
1.	Installation process

    Clone the repo:
    ```sh
    git clone https://github.com/your_username_/Project-Name.git
    ```
    Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
    

3.	API references
    Get a free API Key at https://huggingface.co/.
    Create a .env file 
    ```txt
    HUGGINGFACEHUB_API_TOKEN = "your_token"
    ```
    The hugging face repo id being used: *meta-llama/Meta-Llama-3-8B-Instruct*


# Build and Test
In one terminal:
```sh
cd fastapi 
streamlit run client.py 
```
In another terminal:
```sh
cd fastapi 
uvicorn main:app --reload
```

Upload a resume pdf. 
The recruitment bot will ask a question to the candidate based on their resume (audio and text generated). 
Type in the answer. 
A follow-up question will be asked. 
Continue till the interview is over. 
Total number of questions asked can be set in the fastapi/client.py file (currently set to 2).

The LangGraph workflow is as follows:

![alt text](img/LangGraphWorkflow.jpg)

# Roadmap 
1. Using websockets with whisper model for real time transcription. Integrate real-time speech-to-text functionality: Using the fasterwhisper model
2. Using LangSmith to evaluate the interview? Or an LLM.