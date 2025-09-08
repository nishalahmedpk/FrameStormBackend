from hashlib import new
from typing import TypedDict, Union, Annotated
from langgraph.graph.message import add_messages, AnyMessage

from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv
import os
load_dotenv()


class State(TypedDict,total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    project_id: str
    title: str
    context: str
    blog: str


app = FastAPI()


@app.post("/createProject")
def create_project(projectId: str, title: str, context: str):
    # class State(TypedDict,total=False):
    #     messages: Annotated[list[AnyMessage], add_messages]
    #     project_id: str
    #     title: str
    #     context: str

    state = State(
        project_id=projectId,
        title=title,
        context=context
    )
    return state


@app.post("/generateBlog")
def create_blog(state: State):
    import time
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"for a project titled '{state.get('title')}' with the context '{state.get('context')}', generate a detailed blog post in .Markdown format. Only provide the blog content without any additional commentary or explanations."
    )
    # print(response.text)
    state["blog"] = response.text
    return state,response.text
    # return {"messages": response}

@app.post("/chatblog")
def chat_blog(state: State, message: str):
    from workflow import graph,HumanMessage
    try:
        state["messages"] = state.get("messages", []) + [HumanMessage(content=message+"\nHere is the blog data: "+state.get("blog",""))]
    except Exception as e:
        state["messages"] = [HumanMessage(content=message+"\nHere is the blog data: "+state.get("blog",""))]
        print("what")
    result = graph.invoke(state)
    return result, result["messages"][-1].content
