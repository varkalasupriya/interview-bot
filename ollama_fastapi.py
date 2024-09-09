from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, Response
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import wavio
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, ChatMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace
from typing import Literal
from langgraph.graph import END
import uuid
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Optional
import requests
import json
from langchain import LLMChain, PromptTemplate
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

load_dotenv()

app = FastAPI()

## VARIABLES
TOTAL_QUESTIONS = 6

# token for pocketbase
auth_token = None

model = OllamaLLM(model="llama3.1")

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

parser = PydanticOutputParser(pydantic_object=AskHuman)

template = """Your task is to interview the candidate based on their background information, asking only 5 questions in total. 
Make sure the questions are short and straightforward, without any lengthy explanations.

1. Greet the candidate by their name and ask them to introduce themselves.

2. Ask the candidate to explain one of their projects briefly.

3. Ask the first technical question based on their resume.

4. Ask the second technical question based on their resume.

5. Go through the following information about Centralogic and ask the candidate if they have any questions about the company:
Centralogic was founded in 2010 by Mr. Ajay Navgale, Founder, Director & CEO, and Mr. Sanjay Navgale, Founder & Director. Its services include Project Management, DevOps, Cloud Migration, Cyber Security, and others. Centralogic offers a salary of 3.45 LPA for freshers, with a 2-year bond requirement.

Do not provide feedback, suggestions, or justifications. Only output the next question based on the candidate's responses.

If the candidate asks for clarification or says 'repeat the question,' then rephrase the question in a more understandable way.

"""
# ###################################################################### To Score #############################################################

prompt_template2 = """
Question: {question}
Candidate's Answer: {answer}

Evaluate the candidate's answer based on the question.

The range of score is between 0 and 10, with 0 being the lowest and increasing to 10 which is the highest score.

For answers like "repeat question," "ok," "I don't know," give the lowest score, which is 0.

If the answer responded is contextual but incorrect, based on your own judgment give a score between 4 to 7.

If the answer is contextual and correct based on your own judgement give a score between 8 to 10

If the answer provided by the candidate is inappropriate, off-topic or irrelevant to the question, give a score based on your own judgement in the range 0 to 3.

Your response should strictly contain only the score as a single integer
"""


prompt2 = ChatPromptTemplate.from_template(prompt_template2)

# And a query intended to prompt a language model to populate the data structure.
chain2 = prompt2 | model

def evaluate_answer(question: str, answer: str) -> int:
    result  = chain2.invoke({"question": question, "answer": answer})
    #print(f"this is response from llm:- {result}")
    try:
        score = int(result.strip())
        print(f"Score: {score}")
    except ValueError:
        print("The response did not return a valid integer score.")
        score = 0
    #print(f"from inside score function:- {score}")
    return score


def get_messages_info(messages: list):
    return [SystemMessage(content=template)] + messages

chain = get_messages_info | model

def get_state(messages: list) -> Literal["bye", "ask_human"]:
    if not isinstance(messages[-1], ChatMessage):
        return "ask_human"
    return "info"

def end_result(messages: list):
    return "Thank you for your time."

from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
workflow = MessageGraph()

def ask_human(state: list):
    pass

# Correct way to invoke `chain`
def ask_human2(state: list):
    # Get the messages for the chain
    messages = get_messages_info(state)
    
    # Invoke the chain with the messages
    response = chain.invoke(messages)
    
    # Process the response
    if isinstance(response, str):
        return response
    else:
        # Handle cases where response is not a string
        return "Something went wrong."


# Workflow and graph setup
workflow.add_node("info", ask_human2)
workflow.add_node("bye", end_result)
workflow.add_node("ask_human", ask_human)

workflow.set_entry_point("info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("ask_human", "info")
workflow.add_edge("bye", END)


graph = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def graph_stream(ans: str, msg_num: int, msg_info: list, uid: str, name: str):
    msg = None       
    for output in graph.stream([ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'):
        #print(f"######################################################Output received: {output}")  # Debugging line
        last_message = next(iter(output.values()))
        #print(f"####################################################Last message content: {last_message}")  # Debugging line

        if msg_num >= TOTAL_QUESTIONS * 2:              
            msg = end_result([])
            yield msg
            break

        if "output" in output.keys():
            last_message = output["info"]
        
        if isinstance(last_message, str):
            msg = last_message
        elif hasattr(last_message, 'content'):
            msg = last_message.content
        
        print(f"Message assigned: {msg}")  # Debugging line

        if msg:
            store_interview_data(uid, name, msg_info)
            yield msg
            print(msg_info)


class AnswerModel(BaseModel):
    ans: str
    msg_num: Optional[int] = 0
    msg_info: list
    uid : str
    name : str

@app.post("/langgraph")
async def main(answer: AnswerModel):
    return StreamingResponse(graph_stream(answer.ans, answer.msg_num, answer.msg_info, answer.uid, answer.name), media_type="application/json")

@app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def check_langgraph():
    image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    return Response(content=image_bytes, media_type="image/png")

# PocketBase integration
def get_auth_token():
    global auth_token
    if auth_token is None:
        data={
        "identity": os.getenv("POCKETBASE_USER"),
        "password": os.getenv("POCKETBASE_PASS")
        }
        auth_end_point= "https://dev-pocketbase3.huhoka.com/api/admins/auth-with-password"
        response= requests.post(auth_end_point, json=data)
        decoded_string = response.content.decode('utf-8')
        json_data = json.loads(decoded_string)
        auth_token= json_data["token"]
    return auth_token

def store_interview_data(uid: str, name: str, msg_info: list):
    base_url = 'https://dev-pocketbase3.huhoka.com'
    collection = 'interview_bot'
    endpoint = f'{base_url}/api/collections/{collection}/records'
    score = 0

    latest_ai_message = None
    latest_user_message = None

    for message in reversed(msg_info):
        if message['role'] == 'ai' and latest_ai_message is None:
            latest_ai_message = message['content']
        elif message['role'] == 'user' and latest_user_message is None:
            latest_user_message = message['content']
        
        if latest_ai_message and latest_user_message:
            score = evaluate_answer(latest_ai_message, latest_user_message)
            # print(f"Score is by another outside score llm:- {score}")
            break
    
    data = {
        'uId': uid,
        'candidateName': name,
        'question': latest_ai_message,
        'answer': latest_user_message,
        'score': score
    }

    headers = {
        'Authorization': get_auth_token(),
        'Content-Type': 'application/json'
    }

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 201:
        print('Record created:', response.json())
    else:
        print('Error:', response.status_code, response.text)

# WebSocket for real-time transcription
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-base"
# this is a base model that we are using in here, because of no GPU, when we test it in server then 
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    transcription_file = open("transcription.txt", "a+")
    chunk_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            if data == b"STOP":
                print("Received stop signal from client.")
                break

            chunk_counter += 1
            print(f"Received audio chunk {chunk_counter} from client.")
            audio_filename = "recorded_audio.wav"

            # Save audio chunk
            audio_data = np.frombuffer(data, dtype=np.int16)
            wavio.write(audio_filename, audio_data, rate=16000, sampwidth=2)
            print("audio saved in .wav file")

            # Process audio for transcription
            audio_data_float = audio_data.astype(np.float32) / 32768.0
            result = pipe(audio_data_float, return_timestamps=True, generate_kwargs={"language": "english"})
            transcription_text = result["text"]
            print(f"Transcription for chunk {chunk_counter}: {transcription_text}")

            transcription_file.writelines(transcription_text)
            print(f"Transcription for chunk {chunk_counter} saved to file.")

            # Send transcription back to client
            await websocket.send_text(transcription_text)
            print(f"Transcription for chunk {chunk_counter} sent back to client.")
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        transcription_file.close()
        print("Transcription file closed.")

