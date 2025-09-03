from typing import Annotated, NotRequired, List

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.agents import Tool,create_react_agent,AgentExecutor
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
from langsmith import traceable
from typing import Literal, TypedDict,Sequence,Annotated,Optional,Union
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END,START
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_core.tools import InjectedToolCallId, tool
from pydantic import BaseModel
from langgraph.types import Command 
from langgraph.prebuilt import ToolNode
from datetime import datetime
import uuid

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = init_chat_model("google_genai:gemini-2.0-flash") 

class State(TypedDict,total=False):
    messages: Annotated[list[AnyMessage], add_messages]

from langchain_core.prompts import ChatPromptTemplate

videoeditorprompt = ChatPromptTemplate.from_messages([
    ("system", "Use the tools available to you and the video data to edit the video according the changes required."),("human", "{input}")
])

def Main(state: State):
    llm_1 = videoeditorprompt | (llm.bind_tools([update_state]))
    llm_response = llm_1.invoke(state["messages"])
    return {"messages":llm_response}
