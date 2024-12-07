import sys
import subprocess
import pkgutil
import os

"""PACKAGE INSTALLER"""

#packages
installs = ['langchain_google_community', 'huggingface_hub', 'langchain', 'langchain-community', 'langchain-core', 'gradio', 'torch', 'python-dotenv']
# implement pip as a subprocess:
for package in installs:
    if pkgutil.find_loader(package) is not None:
        print(package + " is installed")
    else:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

"""DOTENV"""

from dotenv import load_dotenv

load_dotenv()

"""IMPORTS"""

import os
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
import torch
from huggingface_hub import login
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent, load_tools, AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import PromptTemplate
import gradio as gr

"""GOOGLE CUSTOM SEARCH TOOL"""

#ENVIRONMENT SETUP
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSD_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#CREATING GOOGLE SEARCH TOOL
search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

"""MODEL"""

#LOGIN
login(token=os.getenv("HGtoken"))

#CREATING MODEL WITH ENDPOINT
llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
local_llm=ChatHuggingFace(llm=llm)

"""EMBEDDING AGENT"""

#PROMPT FOR AGENT
prompt = hub.pull("hwchase17/structured-chat-agent")


#CHAT MEMORY STORAGE
memory = ChatMessageHistory(session_id="session")

#AGENT
agent=create_structured_chat_agent(local_llm,
                                   [tool],
                                   prompt
                                   )
#AGENT_EXECUTOR
agent_executor = AgentExecutor(agent=agent, tools=[tool],
                               handle_parsing_errors=True,
                               max_iterations=10
                               )

#EXECUTABLE AGENT WITH MEMORY
agent_with_chat_history = RunnableWithMessageHistory(agent_executor,
                                                     lambda session_id: memory,
                                                     input_messages_key="input",
                                                     history_messages_key="chat_history"
                                                    )

"""USER INTERFACE"""

#Chatbot funtion
def chatbot_response(choices, from_location, to_location, message):
    input_messages = f"Additional infomation : {message}" + f"(Preference: {choices}, From: {from_location}, To: {to_location})" # chatbot prompt

    if choices == choices[1] or choices == choices[2]:
        input_messages = f"Additional infomation : {message}" + f"(Preference: {choices}, To: {to_location})" # chatbot prompt if 'from_location' is not needed

    answer = agent_with_chat_history.invoke(
        {"input": input_messages},
       config={"configurable": {"session_id": "<foo>"}}
    )

    return answer["output"]

#Chatbot conversation function
def chatbot_response_conversation(message, history):
  answer = agent_with_chat_history.invoke({"input":message},
                                          config={"configurable": {"session_id": "<foo>"}})
  return answer['output']

# Added radio buttons for easy selection
choices = gr.Radio(
    choices=["Traveling to destination", "Accommodation", "Places to visit"],
    label= "Select Your Preference:   " + "*required* "
)

#chatbot UI
chatbot_ui = gr.Interface(
    title="WanderWise",
    fn=chatbot_response,
    inputs=[
        choices,
        gr.Textbox(label="From", placeholder="Enter your location..."),
        gr.Textbox(label="To", placeholder="Enter your destination..."),
        gr.Textbox(label="Preferences", placeholder="Provide any preferences you may have...")
        ],
    outputs="text",

)

#chatbot UI
chatbot_conversation_ui = gr.ChatInterface(chatbot_response_conversation,
                              title="WanderWise Chat")

# Creates an app for support for additional interfaces
app = gr.TabbedInterface(
    [chatbot_ui,chatbot_conversation_ui],
    ["Searchbot","Chatbot"]
)

if __name__ == "__main__":
  app.launch(inbrowser=True, share=False)