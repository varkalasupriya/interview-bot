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
TOTAL_QUESTIONS = 6
 
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
 
if "tts_model" not in st.session_state:
    st.session_state.tts_model = None
 
if st.session_state.tts_model is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts = st.session_state.tts_model
 
gif_placeholder = st.empty()
 
# Function to generate and play audio using xTTS
def generate_and_play_audio(text, speaker_wav, language="en", file_path="output.wav"):
    try:
        # Start video before playing the audio
        st.session_state.is_asking = True
        display_gif(True)  # Bot is asking a question
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        play_audio(file_path)
    except Exception as e:
        st.error(f"Error during audio generation: {e}")
 
# Function to display GIF based on bot state
def display_gif(is_asking):
    if is_asking:
        gif_placeholder.image("new_l.gif", use_column_width=False, width=400)
    else:
        gif_placeholder.image("new_b.gif", use_column_width=False, width=400)
 
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
 
def play_audio(file_path):
    # Start the 'asking' GIF
    display_gif(True)  
   
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
   
    # Start the 'silent' GIF after the audio finishes
    display_gif(False)
 
# Function to start the WebSocket client and send audio chunks
async def send_audio(uri):
    async with websockets.connect(uri) as websocket:
        silence_start = time.time()
        while st.session_state.running:
            audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            await websocket.send(audio.tobytes())
            transcription = await websocket.recv()
            st.session_state.transcription += transcription
 
            if "Thank you." in transcription or not transcription.strip():
                st.session_state.running = False
                break
 
        await websocket.send(b"STOP")
 
# Function to start the WebSocket connection
async def start_websocket():
    st.session_state.running = True
    await send_audio('ws://localhost:8000/ws')
 
# Streamlit UI for uploading PDF
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
 
    l = len(st.session_state.messages)
    asked = l >= (2 * TOTAL_QUESTIONS)
    ans = st.session_state.raw_text if st.session_state.pdf_flag else ""
 
    if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node") or st.session_state.raw_text:
        if not st.session_state.raw_text:
            st.session_state.messages.append({"role": "user", "content": ans})  
       
        response = requests.post('http://localhost:8000/langgraph',
                                 json={
                                     "ans": ans,
                                     "msg_num": len(st.session_state.messages),
                                     "msg_info": st.session_state.messages,
                                     "uid" : st.session_state.uid,
                                     "name": st.session_state.candidate_name
                                 })
 
        st.session_state.messages.append({"role": "ai", "content": response.text})
        st.session_state.question_asked = True
 
        # Generate and play AI response in cloned voice
        generate_and_play_audio(response.text, "output.wav", language="en", file_path="outputs/ai_audio.wav")
 
        st.session_state.raw_text = ""
        asyncio.run(start_websocket())
        st.rerun()
   
    @st.cache_data
    def cached_get_pdf_text(pdf_docs):
        return get_pdf_text(pdf_docs)
 
    if st.session_state.transcription:
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
            generate_and_play_audio(response.text, "output.wav", language="en", file_path="outputs/ai_audio.wav")
            st.session_state.transcription = ""
            asyncio.run(start_websocket())
            st.rerun()
        else:
            st.error("Failed to get response from the server.")
else:
    st.error('Failed to retrieve data')