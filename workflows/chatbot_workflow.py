from itertools import count
import os
from unittest.mock import Base
from cv2 import add
from langgraph import graph
from langgraph.graph import  START, StateGraph,START, END, add_messages
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
import operator

# load env
load_dotenv() # Loads variables from .env
os.environ["HUGGINGFACEHUB_API_TOKEN"]



# uising hugging face endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define node
def chatbot_node(state: ChatState, config: RunnableConfig) -> ChatState:
    response = chat_model.invoke(state["messages"])
    return {"messages": [response]}  # only return new messages


# build workflow
agent_builder  = StateGraph(ChatState)
# add nodes
agent_builder.add_node('chatbot_node', chatbot_node)
# add edges to connet the nodes
agent_builder.add_edge(START, 'chatbot_node')
agent_builder.add_edge("chatbot_node",END)
# compile the agent

# Invoke


checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot.")
        break
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    print(f"Chatbot: {response['messages'][-1].content}")



print(list(agent.get_state_history(config)))

