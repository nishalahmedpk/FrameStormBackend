from typing import Dict, Union

from fastapi import FastAPI

from videoworkflow import graph, State

from pydantic import BaseModel

app = FastAPI()

conversation_states: Dict[str, dict] = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Viral Video Maker API"}

@app.get("/generate_video")
def generate_video(project_name: str,description: str, generate_video: bool):
    """
    Endpoint to generate a viral video based on the provided description.
    
    :param project_name: Name of the project for video generation.
    :param description: Description for the viral video.
    :param generate_video: Flag to indicate whether to generate a video.
    :return: Result of the video generation process.
    """

    import os
    import shutil

    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    initial_state = State(description=description, generate_video=generate_video, project_name=project_name)
    
    # Invoke the graph with the initial state
    result = graph.invoke(initial_state)
    
    return {"graph_state": result,"path":os.path.abspath(project_name)}

@app.get("/generate_image")
def generate_image(prompt: str, project_name: str):
    """
    Endpoint to generate an image based on the provided prompt.
    
    :param prompt: Text prompt for image generation.
    :param size: Size of the generated image (default is "1024*1024").
    :return: URL of the generated image.
    """
    from http import HTTPStatus
    from urllib.parse import urlparse, unquote
    from pathlib import PurePosixPath
    import requests
    from dashscope import ImageSynthesis
    import os
    import dashscope

    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)
    
    print('please wait...')
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    # prompt = "A flower shop with exquisite windows, a beautiful wooden door, and flowers on display"

    print('----sync call, please wait a moment----')
    rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                            model="wan2.2-t2i-flash",
                            prompt=prompt,
                            n=1,
                            size='864*1080')
    print('response: %s' % rsp)
    if rsp.status_code == HTTPStatus.OK:
        # Save the image in the current directory
        for result in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            file_name = os.path.join(project_name, file_name)
            with open('./%s' % file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
            return {"image_path": os.path.abspath(file_name)}
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))
        return {"image_path": None}

@app.get("/generate_blog")
def generate_blog(prompt: str, project_name: str):
    """
    Endpoint to generate a blog post based on the provided prompt.
    
    :param prompt: Text prompt for blog generation.
    :return: Generated blog content.
    """
    from videogeneration import BlogGenerator

    blog_post = BlogGenerator(prompt)
    return {"blog_post": blog_post}


class ChatRequest(BaseModel):
    project_name: str
    question: str

@app.post("/chat_with_video")
def chat_with_video(request: ChatRequest):
    """
    Endpoint to chat with the generated video content with persistent state.
    
    :param request: ChatRequest containing project_name and question
    :return: Response to the question based on the video content and persistent conversation
    """
    from videoworkflow import chatgraph
    import os

    # Retrieve or initialize state for this project
    state = conversation_states.get(request.project_name, {"messages": [], "project_name": request.project_name})

    # Append the new user message to state
    state["messages"].append({"role": "user", "content": request.question})

    # Invoke the chat graph with the updated state
    result = chatgraph.invoke(state, request.question)

    # Update the stored state
    conversation_states[request.project_name] = state

    # Return the response and path for frontend usage
    return {
        "graph_state": result,
        "conversation": state["messages"],
        "project_path": os.path.abspath(request.project_name)
    }
