#!/usr/bin/env python3
"""
CrewAI Document Extraction Launch Script

This script runs the CrewAI orchestrator to extract document embeddings
from PDF files in the ./docs directory and store them in the vector database.

Usage:
    # Run locally:
    python run_crewai_extraction.py
    
    # Run in Docker Compose:
    docker-compose up crewai-extraction

Prerequisites:
    1. Set your OpenAI API key in one of these ways:
       - Environment variable: export OPENAI_API_KEY=your-key
       - .env file: OPENAI_API_KEY=your-key
       - Docker Compose: OPENAI_API_KEY=your-key in .env file
    
    2. Ensure PostgreSQL database is running:
       docker-compose up postgres -d
    
    3. Place PDF files in the ./docs directory
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from tools import DocextractTool
from langchain_openai import OpenAI

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Load environment variables (from .env file or environment)
    load_dotenv()
    
    # Check if API key is set (from environment or .env file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ Error: OPENAI_API_KEY not properly set!")
        print("   Please set your OpenAI API key in one of these ways:")
        print("   - Environment variable: export OPENAI_API_KEY=your-key")
        print("   - .env file: OPENAI_API_KEY=your-key")
        print("   - Docker Compose: OPENAI_API_KEY=your-key in .env file")
        return False
    
    # Check if docs directory exists and has PDF files
    docs_dir = Path("./docs")
    if not docs_dir.exists():
        print("❌ Error: ./docs directory not found!")
        print("   Please create the docs directory and add PDF files to process.")
        return False
    
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ Error: No PDF files found in ./docs directory!")
        print("   Please add PDF files to the ./docs directory.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s) to process")
    print("✅ OpenAI API key configured")
    print("✅ All prerequisites met!")
    return True

def run_crewai_extraction():
    """Run the CrewAI document extraction process."""
    print("\n🚀 Starting CrewAI Document Extraction...")
    print("=" * 50)
    
    try:
        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        
        # Create the agent
        print("🤖 Creating Document Processor Agent...")
        agent = Agent(
            role="Document Processor",
            goal="Extract and store document embeddings for semantic search",
            backstory="You are an expert in processing documents and preparing them for semantic search.",
            tools=[DocextractTool()],
            llm=llm,
            verbose=True
        )
        
        # Create the task
        print("📋 Creating extraction task...")
        task = Task(
            description="Extract embeddings from PDF files in the './docs' directory and store them in the vector database.",
            expected_output="A summary of processed files and the number of chunks stored.",
            agent=agent
        )
        
        # Create the crew
        print("👥 Assembling CrewAI team...")
        crew = Crew(agents=[agent], tasks=[task])
        
        # Execute the crew
        print("\n⚡ Executing CrewAI extraction process...")
        print("-" * 50)
        result = crew.kickoff()
        
        # Display results
        print("\n" + "=" * 50)
        print("🎉 CREWAI EXTRACTION COMPLETED!")
        print("=" * 50)
        print("\n📊 EXTRACTION RESULTS:")
        print("-" * 30)
        print(result)
        print("\n✅ Document extraction process finished successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during CrewAI extraction: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   - Ensure PostgreSQL database is running: docker-compose up postgres -d")
        print("   - Check your OpenAI API key is valid and has credits")
        print("   - Verify PDF files are readable and not corrupted")
        return False

def main():
    """Main function to orchestrate the document extraction process."""
    print("📚 CrewAI Document Extraction Launcher")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run the extraction
    success = run_crewai_extraction()
    
    if success:
        print("\n🎯 Next steps:")
        print("   - Access the query interface: http://localhost:50505")
        print("   - Or start the full application: docker-compose up -d")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
