from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import os
import shutil

from videoworkflow import graph, chatgraph
from videoworkflow import BlogGenerator
from dashscope import ImageSynthesis
import dashscope
import requests
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath

app = FastAPI()

# Persistent conversation states per project
conversation_states: Dict[str, dict] = {}

# ------------------------------
# State class
# ------------------------------
class State(dict):
    messages: list = []
    final_video: str = ""
    project_name: str = "Viral Video Maker"
    description: str = ""
    generate_video: bool = False
    prompts: list = []
    script: str = ""
    voiceover: str = ""
    cuts: list = []
    audio_file: str = ""
    video_files: list = []
    audio_duration: float = 0.0
    blog: str = ""

# ------------------------------
# Generate Video
# ------------------------------
@app.post("/generate_video")
def generate_video_api(project_name: str, description: str, generate_video: bool):
    """
    Generate a viral video based on description.
    Persistent state is stored per project.
    """
    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    # Retrieve existing state or create a new one
    state = conversation_states.get(project_name)
    if not state:
        state = State()
        state["project_name"] = project_name
        state["description"] = description
        state["generate_video"] = generate_video
        state["messages"] = []
        conversation_states[project_name] = state
    else:
        # Update description and generate_video flag if needed
        state["description"] = description
        state["generate_video"] = generate_video

    # Invoke the video graph
    result = graph.invoke(state)

    state["messages"] = result.get("messages", [])
    state["final_video"] = result.get("final_video", "")
    state["cuts"] = result.get("cuts", [])
    state["video_files"] = result.get("video_files", [])
    state["audio_file"] = result.get("audio_file", "")
    state["audio_duration"] = result.get("audio_duration", 0.0)
    state["voiceover"] = result.get("voiceover", "")
    conversation_states[project_name] = result

    return {
        "graph_state": result,
        "project_path": os.path.abspath(project_name),
        "state": state
    }

# ------------------------------
# Generate Image
# ------------------------------
@app.post("/generate_image")
def generate_image(prompt: str, project_name: str):
    """
    Generate an image based on the provided prompt.
    Persistent state is stored per project.
    """
    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    rsp = ImageSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.2-t2i-flash",
        prompt=prompt,
        n=1,
        size='864*1080'
    )

    if rsp.status_code == HTTPStatus.OK:
        for result_item in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result_item.url).path)).parts[-1]
            file_path = os.path.join(project_name, file_name)
            with open(file_path, "wb") as f:
                f.write(requests.get(result_item.url).content)

            # Update state
            state = conversation_states.get(project_name, State())
            state.setdefault("prompts", []).append(prompt)
            conversation_states[project_name] = state

            return {"image_path": os.path.abspath(file_path), "state": state}
    else:
        return {"image_path": None, "state": conversation_states.get(project_name)}

# ------------------------------
# Generate Blog
# ------------------------------
@app.post("/generate_blog")
def generate_blog(prompt: str, project_name: str):
    """
    Generate a blog post based on prompt and save in state.
    """
    blog_post = BlogGenerator(prompt)

    # Update state
    state = conversation_states.get(project_name, State())
    state["blog"] = blog_post
    conversation_states[project_name] = state

    return {"blog_post": blog_post, "state": state}

# ------------------------------
# Chat with Video (Persistent)
# ------------------------------
class ChatRequest(BaseModel):
    project_name: str
    question: str

@app.post("/chat_with_video")
def chat_with_video(request: ChatRequest):
    """
    Chat with the video content using persistent state.
    """
    # Retrieve or initialize state for this project
    state = conversation_states.get(request.project_name, State())
    print(state)
    state["project_name"] = request.project_name
    state.setdefault("messages", [])
    state["input"] = request.question
    state["messages"].append({"role": "user", "content": request.question})

    # Invoke the chat graph
    result = chatgraph.invoke(state)

    # Update persistent state
    conversation_states[request.project_name] = state

    return {
        "graph_state": result,
        "conversation": state["messages"],
        "project_path": os.path.abspath(request.project_name),
        "state": state
    }
