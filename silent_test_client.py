# in this file we used xtts model in order to get the cloned voice from Ai
 
import streamlit as st
import requests
import asyncio
import websockets
import time
import sounddevice as sd
from dotenv import load_dotenv
from pypdf import PdfReader
import pyaudio
import wave
import torch
from TTS.api import TTS
import uuid
 
## VARIABLES
TOTAL_QUESTIONS = 10
 
## Initialize session state attributes if they don't exist
if "pdf_flag" not in st.session_state:
    st.session_state.pdf_flag = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "running" not in st.session_state:
    st.session_state.running = False
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())
 
# Initialize xTTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
 
# Function to generate and play audio using xTTS
def generate_and_play_audio(text, speaker_wav, language="en", file_path="output.wav"):
    try:
        wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        play_audio(file_path)
    except Exception as e:
        st.error(f"Error during audio generation: {e}")
 
 
## Read the pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
def get_name(text):
    lines = text.split('\n')
    candidate_name = lines[0].strip()
    return candidate_name
 
# Function to play audio using PyAudio
def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()
 
# Helper function to check for silence in the audio
def check_silence(audio, threshold=500):
    return max(audio) < threshold
 
# Function to start the WebSocket client and send audio chunks
async def send_audio(uri):
    async with websockets.connect(uri) as websocket:
        silence_start = time.time()
        while st.session_state.running:
            audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
 
            if check_silence(audio):
                if time.time() - silence_start > 5:
                    st.session_state.running = False
                    break
            else:
                silence_start = time.time()
 
            await websocket.send(audio.tobytes())
            transcription = await websocket.recv()
            st.session_state.transcription += transcription
 
        await websocket.send(b"STOP")
 
# Function to start the WebSocket connection
async def start_websocket():
    st.session_state.running = True
    await send_audio('ws://localhost:8000/ws')
 
# Streamlit UI for uploading PDF and displaying chat
response = requests.get('http://localhost:8000/')
if response.status_code == 200:
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.session_state.raw_text = get_pdf_text(pdf_docs)
                st.session_state.candidate_name = get_name(st.session_state.raw_text)
                if st.session_state.raw_text:
                    st.success("Done")
                    st.session_state.pdf_flag = True
 
    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
 
    display_chat()
 
    l = len(st.session_state.messages)
    # TOTAL_QUESTIONS = 10  # Adjust as needed
 
    if l >= (2 * (TOTAL_QUESTIONS)):
        asked = True
    else:
        asked = False
 
    ans = st.session_state.raw_text if st.session_state.pdf_flag else ""
 
    if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node") or st.session_state.raw_text:
        if not st.session_state.raw_text:
            with st.chat_message("user"):
                st.markdown(ans)
                st.session_state.messages.append({"role": "user", "content": ans})  
       
        response = requests.post('http://localhost:8000/langgraph',
                                 json={
                                     "ans": ans,
                                     "msg_num": len(st.session_state.messages),
                                     "msg_info": st.session_state.messages,
                                     "uid" : st.session_state.uid,
                                     "name": st.session_state.candidate_name
                                 })
        with st.chat_message("ai"):
            st.markdown(response.text)
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True
 
            # Generate and play AI response in cloned voice
            generate_and_play_audio(response.text, "prajwal.wav", language="en", file_path="outputs/ai_audio.wav")
 
        st.session_state.raw_text = ""  
        asyncio.run(start_websocket())
        st.rerun()
 
    if st.session_state.transcription:
        with st.chat_message("user"):
            st.markdown(st.session_state.transcription)
            st.session_state.messages.append({"role": "user", "content": st.session_state.transcription})
        response = requests.post('http://localhost:8000/langgraph',
                                json={
                                    "ans": st.session_state.transcription,
                                    "msg_num": len(st.session_state.messages),
                                    "msg_info": st.session_state.messages,
                                    "uid" : st.session_state.uid,
                                    "name": st.session_state.candidate_name
                                })
        if response.status_code == 200:
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True
            with st.chat_message("ai"):
                st.markdown(response.text)
                generate_and_play_audio(response.text, "prajwal.wav", language="en", file_path="outputs/ai_audio.wav")
                st.session_state.transcription = ""
                asyncio.run(start_websocket())
                st.rerun()
        else:
            st.error("Failed to get response from the server.")
       
else:
    st.error('Failed to retrieve data')