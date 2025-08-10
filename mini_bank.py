# from agents import Agent, Runner, OpenAIChatCompletionsModel, input_guardrail,set_tracing_disabled
# from openai import AsyncOpenAI
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from agents import (InputGuardrail,
#                     RunContextWrapper,
#                     TResponseInputItem,
#                     GuardrailFunctionOutput,
#                     InputGuardrailTripwireTriggered
#                     )
# import chainlit as cl 

# load_dotenv()
# set_tracing_disabled(disabled=True)

# gemini_api_key=os.getenv("GEMINI_API_KEY")

# client= AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model= OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=client,
# )
# class Outputbank(BaseModel):
#     is_banking_related: bool
#     reasoning: str

# input_guardrials_agents= Agent(
#     name="Input Guardrails Checker",
#     instructions="Check if the user question is banking related,retrun true,if the quetion is not related to banking, return flase",
#     model=model,
#     output_type=Outputbank
# )

# @input_guardrail
# async def input_guardrail_functions(
#     ctx: RunContextWrapper,
#     agent: Agent,
#     input: str | list[TResponseInputItem]
# )->GuardrailFunctionOutput:
#     result= await Runner.run(
#        input_guardrials_agents,
#        input
#     )
#     return GuardrailFunctionOutput(
#         output_info=result.final_output,
#         tripwire_triggered=not result.final_output.is_banking_related,
#     )

# trige_agent= Agent(
#     name="Banking Agent",
#     instructions="you are a banking agent you only respond about banking related question",
#     model=model,
#     input_guardrails=[input_guardrail_functions]
# )

# @cl.on_chat_start
# async def on_chat_start():
#     await cl.Message(content="I'm ready to assist you").send()

# @cl.on_message
# async def on_message(message:cl.Message):
#     try:
#         result= await Runner.run(
#             trige_agent,
#             input=message.content
#         )
#         await cl.Message(content=result.final_output).send()
#     except InputGuardrailTripwireTriggered:
#         await cl.Message(content="Please Ask Banking Questions").send()
# bank_agent_project.py

from agents import Agent, Runner, OpenAIChatCompletionsModel, input_guardrail, output_guardrail, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import chainlit as cl
import os

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

# Gemini API setup (OpenAI-compatible mode)
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Input Guardrail Schema
class BankInputCheck(BaseModel):
    is_banking_related: bool
    reasoning: str

input_guardrails_agent = Agent(
    name="Input Guardrail Checker",
    instructions="Check if the question is about banking. Return true if it is; false otherwise.",
    model=model,
    output_type=BankInputCheck
)

@input_guardrail
async def validate_banking_input(ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrails_agent, input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_banking_related
    )

# Output Guardrail Schema
class OutputSafetyCheck(BaseModel):
    is_safe: bool
    reasoning: str

output_guardrails_agent = Agent(
    name="Output Guardrail Checker",
    instructions="Check if the agent's response contains personal or unsafe information. Return false if unsafe.",
    model=model,
    output_type=OutputSafetyCheck
)

@output_guardrail
async def validate_output_safety(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrails_agent, output)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_safe
    )

# Handoff Agents
account_agent = Agent(
    name="Account Agent",
    instructions="You are an account management agent. Answer only about account balances, statements, and account details.",
    model=model
)

loan_agent = Agent(
    name="Loan Agent",
    instructions="You are a loan management agent. Answer only about loans, EMI schedules, and interest rates.",
    model=model
)

# Main Banking Agent
banking_agent = Agent(
    name="Banking Agent",
    instructions="You are a banking assistant. Only answer banking-related queries. Route to correct department.",
    model=model,
    input_guardrails=[validate_banking_input],
    output_guardrails=[validate_output_safety],
    handoffs=[account_agent, loan_agent]
)

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="💳 Welcome to the Banking Assistant! How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        result = await Runner.run(banking_agent, input=message.content)
        await cl.Message(content=str(result.final_output)).send()
    except InputGuardrailTripwireTriggered:
        await cl.Message(await cl.Message(content="Please Ask Banking Questions").send())
