import argparse
import os

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from env import ANTHROPHIC_API


def get_ollama_model_as_tool(model='medbot-raw:latest'):
    # medical llm as tool
    prompt = ChatPromptTemplate.from_messages([("human", "The following are the symptoms of my patient: {symptoms}.")])

    llm = ChatOllama(model=model)

    chain = prompt | llm | StrOutputParser()

    return chain.as_tool(name='medbot', description='Use this tool when the model is given medical symptoms related to diseases. Do not call this tool if the user question does not include a description of the symtom.')

def get_agent_executor():
    anthropic_env()

    agent_model = ChatAnthropic(model='claude-3-sonnet-20240229')

    tools = [get_ollama_model_as_tool()]

    # memory = MemorySaver()

    agent_executor = create_react_agent(agent_model, tools)

    return agent_executor



def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def anthropic_env():
    # Check if ANTHROPIC_API_KEY exists
    if 'ANTHROPIC_API_KEY' not in os.environ:
        # Set the environmental variable
        os.environ['ANTHROPIC_API_KEY'] = ANTHROPHIC_API


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query')

    args = parser.parse_args()

    query = args.query

    agent = get_agent_executor()

    # Use the agent
    # config = {"configurable": {"thread_id": "abc1231"}}

    print_stream(agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values"))



if __name__ == '__main__':
    main()
