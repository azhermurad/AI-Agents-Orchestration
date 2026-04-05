import json
import operator
import os
from xml.parsers.expat import model
from langgraph.graph import START, StateGraph, START, END
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field


# load env
load_dotenv()  # Loads variables from .env


# uising hugging face endpoint
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     provider="auto",  # let Hugging Face choose the best provider for you
# )

# chat_model = ChatHuggingFace(llm=llm)


from langchain_groq import ChatGroq

chat_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1,
    max_tokens=512,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "feedback_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "feedback": {
                        "type": "string",
                        "description": "Short textual feedback or summary",
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "Rating score between 0 and 5",
                    },
                },
                "required": ["feedback", "score"],
                "additionalProperties": False,
            },
        },
    },
)


# define reducer function for scores list merging
def score_list_adder_reducer(old, new):
    print(f"old: {old}, new: {new}")
    return old + new


class InputState(TypedDict):
    eassy: str


class EassyEvaluationState(TypedDict):
    clarity_feedback: str
    depth_feedback: str
    language_feedback: str
    scores: Annotated[list[int], score_list_adder_reducer]
    final_score: float
    final_feedback: str


class EvaluationResult(BaseModel):
    """A movie with details."""

    feedback: str = Field(description="feedback for the eassy")
    score: int = Field(description="score for the eassy out of 5", ge=0, le=5)


model_with_structure = chat_model
import json
# define node
def evaluate_eassy_on_clarity(state: InputState) -> EassyEvaluationState:
    eassy = state["eassy"]
    prompt = f"""You are a helpful assistant. Please evaluate the following eassy based on the following criteria: 
    1. Clarity of Thought (1–5)
        - Logical flow
        - Clear ideas
        - Easy to understand
    Provide feedback for  criterion and assign a score from 1 to 5 for  criterion. Eassy: {eassy}
    """
    response = model_with_structure.invoke(prompt)
    response = json.loads(response.content)
    return {"clarity_feedback": response['feedback'], "scores": [response['score']]}


def evaluate_eassy_on_Depth_Analysis(state: InputState) -> EassyEvaluationState:
    eassy = state["eassy"]
    prompt = f"""You are a helpful assistant. Please evaluate the following eassy based on the following criteria: 
    1. Depth of Analysis (1–5)
    - Critical thinking
    - Strong reasoning
    - Use of arguments
        Provide feedback for  criterion and assign a score from 1 to 5 for  criterion. Eassy: {eassy}
    """

    response = model_with_structure.invoke(prompt)
    response = json.loads(response.content)
    return {"depth_feedback": response['feedback'], "scores": [response['score']]}


def evaluate_eassy_on_Language_Quality(state: InputState) -> EassyEvaluationState:
    eassy = state["eassy"]
    prompt = f"""You are a helpful assistant. Please evaluate the following eassy based on the following criteria: 
    1. Language Quality (1–5)
    - Grammar
    - Vocabulary
    - Sentence structure
        Provide feedback for  criterion and assign a score from 1 to 5 for  criterion. Eassy: {eassy}
    """
    response = model_with_structure.invoke(prompt)
    response = json.loads(response.content)
    return {"language_feedback": response['feedback'], "scores": [response['score']]}



# finial node to aggregate the scores and feedback
def aggregate_eassy_evaluation(state: EassyEvaluationState) -> EassyEvaluationState:
    total_score = state["scores"]
    final_score = sum(total_score ) # since we have 3 criteria
    return {"final_score": final_score, "final_feedback": "final_feedback"}

    # build workflow


agent_builder = StateGraph(state_schema=EassyEvaluationState, input_schema=InputState)
# add nodes
agent_builder.add_node("clarity", evaluate_eassy_on_clarity)
agent_builder.add_node("depth_analysis", evaluate_eassy_on_Depth_Analysis)
agent_builder.add_node("language_quality", evaluate_eassy_on_Language_Quality)
agent_builder.add_node("aggregate", aggregate_eassy_evaluation)


# add edges to connet the nodes
agent_builder.add_edge(START, "clarity")
agent_builder.add_edge(START, "depth_analysis")
agent_builder.add_edge(START, "language_quality")

agent_builder.add_edge("clarity", "aggregate")
agent_builder.add_edge("depth_analysis", "aggregate")
agent_builder.add_edge("language_quality", "aggregate")

agent_builder.add_edge("aggregate", END)


# compile the agent
# agent = agent_builder.compile()
# Invoke

eassy = """
Technology has changed the way people communicate. However, it has also reduced face-to-face interaction.
This is because people rely more on digital platforms. Therefore, the impact on social skills is significant.
"""
agent = agent_builder.compile()
response = agent.invoke({"eassy": eassy})


print(response)
