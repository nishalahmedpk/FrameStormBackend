from typing import TypedDict, Union, Annotated
from langgraph.graph.message import add_messages, AnyMessage

from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv
import os
load_dotenv()


class State(TypedDict,total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    title: str
    context: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Working My Man": "World"}


@app.post("/generateBlog")
def create_blog(state: State):
    import time
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"for a project titled '{state.get('title')}' with the context '{state.get('context')}', generate a detailed blog post in .Markdown format."
    )
    print(response.text)
    return {"messages": response}
