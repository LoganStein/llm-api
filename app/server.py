from dotenv import load_dotenv
from os import getenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain.agents import AgentType

load_dotenv()

openai_api_key = getenv("OPENAI_API_KEY")

# Router

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant agent."""

    datasource: Literal["sql_agent", "general_qna_agent"] = Field(
        ...,
        description="""Given a user question choose to route it to the general_qna_agent who can handle 
        general conversation or a sql_agent who can query the client database""",
    )

llm = ChatOpenAI(model="gpt-4", temperature=0.9)
structured_llm_router = llm.with_structured_output(RouteQuery)

# prompt
system = """You are an expert at routing a user question to the most relevant agent."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{question}"),
    ]
)

@chain
def custom_chain (question: str) -> str:
    # call the llms here
    return llm.invoke(question)

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse(url="/docs")

add_routes(app, custom_chain, path="/openai")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)