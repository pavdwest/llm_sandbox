from typing import List

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

import config


def get_my_name(*args, **kwargs) -> str:
    return 'Guy McMan'


def weather_tool(*args, **kwargs) -> str:
    return 'warm and sunny from 08:00 to 15:00, cold and rainy after 15:00'


def owned_clothes_tool(*args, **kwargs) -> List[str]:
    return [
        'jacket',
        'shorts',
        'jorts',
        'jeans',
        'chinos',
        'button-up shirt',
        't-shirt',
        'flip-flops',
        'raincoat',
        'speedo',
        'sneakers',
        'boots',
        'mankini',
        'tuxedo',
        'suit and tie',
        'leotard',
        'umbrella',
    ]

def event_details(*args, **kwargs) -> str:
    # return 'The event starts tomorrow at 11am and ends at 2pm. It is an informal event.'
    # return 'The event starts tomorrow at 5pm and ends at 7pm. It is an informal event.'
    return 'The event is from 16:00 to 19:00 tomorrow. It is a formal event.'


# Load the language model
llm = OpenAI(
    temperature=0,
    openai_api_key=config.OPENAI_API_KEY,
)

# Tools
weather_tool_ = Tool(
    name='weather',
    func=weather_tool,
    description="Retrieves tomorrow's weather.",
)

owned_clothes_tool_ = Tool(
    name='clothes',
    func=owned_clothes_tool,
    description="Lists all the clothes and accessories that I own"
)

event_details_tool_ = Tool(
    name='event',
    func=event_details,
    description='Get details of the event I will be attending tomorrow.'
)

tools = [
    weather_tool_,
    owned_clothes_tool_,
    event_details_tool_,
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# agent.run("Using the tools at your disposal, pick the most appropriate clothes for me to wear tomorrow between 10 and 11am.")
# agent.run("Recommend the most appropriate clothes for me to wear tomorrow evening.")
# agent.run("Using the tools at your disposal, pick the most appropriate clothes for me to wear tomorrow between 11am and 8pm. I can only take one set of clothes.")
agent.run("Using as many of the available tools at your disposal, recommend the most appropriate clothes for me to wear to tomorrow's event.")
# agent.run("Should I take an umbrella to tomorrow's event?")

