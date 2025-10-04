from crewai import Agent, Task, Crew
from tools import DocextractTool
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

agent = Agent(
    role="Document Processor",
    goal="Extract and store document embeddings for semantic search",
    backstory="You are an expert in processing documents and preparing them for semantic search.",
    tools=[DocextractTool()],
    llm=llm,
    verbose=True
)

task = Task(
    description="Extract embeddings from PDF files in the './docs' directory and store them in the vector database.",
    expected_output="A summary of processed files and the number of chunks stored.",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result)