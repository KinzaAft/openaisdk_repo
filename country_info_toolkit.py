from agents import Agent,Runner,RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
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
# for capital of the country
@function_tool
def get_capital_country(country: str) -> str:
    capitals = {
        "pakistan": "Islamabad",
        "india": "New Delhi",
        "japan": "Tokyo",
        "germany": "Berlin",
        "france": "Paris",
        "canada": "Ottawa",
        "australia": "Canberra",
        "united states": "Washington, D.C.",
        "united kingdom": "London",
        "italy": "Rome",
        "russia": "Moscow",
        "brazil": "Brasília",
        "south africa": "Pretoria",
        "turkey": "Ankara",
        "afghanistan": "Kabul",
        "bangladesh": "Dhaka",
        "iran": "Tehran",
        "iraq": "Baghdad",
        "saudi arabia": "Riyadh",
        "indonesia": "Jakarta",
        "egypt": "Cairo",
        "mexico": "Mexico City",
        "thailand": "Bangkok",
        "nepal": "Kathmandu"
    }
    return capitals.get(country.lower(), "Capital not found.") #get .lower method is used because python is case sensitve if the user input is in capital or in other form it will return none

# for language of the country
@function_tool
def get_language_country(country: str) -> str:
    languages = {
        "pakistan": "Urdu",
        "india": "Hindi",
        "japan": "Japanese",
        "germany": "German",
        "france": "French",
        "canada": "English and French",
        "australia": "English",
        "china": "Mandarin Chinese",
        "united states": "English",
        "united kingdom": "English",
        "italy": "Italian",
        "russia": "Russian",
        "brazil": "Portuguese",
        "south africa": "11 official languages (including Zulu, Xhosa, Afrikaans, and English)",
        "turkey": "Turkish",
        "afghanistan": "Pashto and Dari",
        "bangladesh": "Bengali",
        "iran": "Persian (Farsi)",
        "iraq": "Arabic and Kurdish",
        "saudi arabia": "Arabic",
        "indonesia": "Indonesian (Bahasa Indonesia)",
        "egypt": "Arabic",
        "mexico": "Spanish",
        "thailand": "Thai",
        "nepal": "Nepali"
    }
    return languages.get(country.lower(), "Language not found.")

# for population of the country
@function_tool
def get_population_country(country: str) -> str:
    populations = {
        "pakistan": "240 million",
        "india": "1.4 billion",
        "japan": "125 million",
        "germany": "83 million",
        "france": "67 million",
        "canada": "39 million",
        "australia": "26 million",
        "china": "1.4 billion",
        "united states": "331 million",
        "united kingdom": "67 million",
        "italy": "59 million",
        "russia": "144 million",
        "brazil": "215 million",
        "south africa": "60 million",
        "turkey": "85 million",
        "afghanistan": "41 million",
        "bangladesh": "173 million",
        "iran": "89 million",
        "iraq": "44 million",
        "saudi arabia": "36 million",
        "indonesia": "276 million",
        "egypt": "110 million",
        "mexico": "126 million",
        "thailand": "70 million",
        "nepal": "30 million"
    }
    return populations.get(country.lower(), "Population not found.")

orchestrator = Agent(
    name="Country Info Bot",
    instructions=(
        "You are a helpful assistant. When given a country name, use the tools to find "
        "its capital city, official language, and population. Combine and present them clearly."
    ),
    tools=[get_capital_country, get_language_country,get_population_country]
)

# this code can be used for terminal based answer
# result = Runner.run_sync(
#     agent,
#     input="Hey,How can i help you?",
#     run_config=config
# )

# print(result.final_output)

# This code is for chainlit


@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip() #.strip is use to remove any extra space from the message
    result = await Runner.run(orchestrator, input=user_input, run_config=config)
    await cl.Message(content=result.final_output).send()