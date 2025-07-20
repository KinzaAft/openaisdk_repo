from agents import Agent,Runner,RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
import os
import chainlit as cl
from dotenv import load_dotenv
load_dotenv()

gemini_api_key= os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled=True
)

agent1 = Agent(
    name="Suggestion Expert",
    instructions="Act as a frined to know about person mood then if he is happy ask him more what is the thing he is happy about"

)
agent2 = Agent(
    name="sad Agent ",
    instructions="Suggest person how to stay calm in strees or tell them excercise to overcome their sadness but make sure to not gave more than 6 line"
)
trigeAgent= Agent(
    name="Trige Agent",
    handoffs=[agent1,agent2]
)
# this code can be used for terminal based answer
# result = Runner.run_sync(
#     agent,
#     input="Hey,How can i help you?",
#     run_config=config
# )

# print(result.final_output)

# This code is for chainlit
@cl.on_chat_start
async def handle_chat_start():
    await cl.Message(content="Hey! How can I help you today?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(
        trigeAgent,
        input=message.content,
        run_config=config,
        # starting_agent= agent1,
    )
    await cl.Message(content=result.final_output).send()
