from crewai import Agent, Task, Crew
from tools import DocextractTool
from langchain_openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your key
llm = OpenAI(model="gpt-3.5-turbo")  # Or another model

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