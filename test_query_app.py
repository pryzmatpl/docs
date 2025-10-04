#!/usr/bin/env python3
"""
Test script for the semantic knowledge base query application.
This script tests the basic functionality without requiring the full Docker setup.
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_query_app():
    """Test the query application endpoints."""
    base_url = "http://localhost:50505"
    
    print("🧪 Testing Semantic Knowledge Base Query Application")
    print("=" * 60)
    
    # Test 1: Check if the app is running
    print("\n1. Testing if the application is running...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Application is running and accessible")
        else:
            print(f"❌ Application returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Application is not accessible: {e}")
        print("💡 Make sure to run: docker-compose up -d")
        return False
    
    # Test 2: Test query endpoint
    print("\n2. Testing query endpoint...")
    test_query = "What is machine learning?"
    try:
        response = requests.post(
            f"{base_url}/query",
            data={"query": test_query},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful: '{test_query}'")
            print(f"   Found {data.get('total_results', 0)} results")
            if data.get('results'):
                print(f"   Top result similarity: {data['results'][0].get('similarity_score', 0):.2%}")
        else:
            print(f"❌ Query failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Query request failed: {e}")
    
    # Test 3: Test history endpoint
    print("\n3. Testing history endpoint...")
    try:
        response = requests.get(f"{base_url}/history", timeout=5)
        if response.status_code == 200:
            print("✅ History endpoint accessible")
        else:
            print(f"❌ History endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ History request failed: {e}")
    
    # Test 4: Test ingestion endpoint
    print("\n4. Testing ingestion endpoint...")
    try:
        response = requests.post(f"{base_url}/ingest", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Ingestion endpoint accessible")
            print(f"   Message: {data.get('message', 'No message')}")
        else:
            print(f"❌ Ingestion failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Ingestion request failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed!")
    print("\n📋 Next steps:")
    print("   1. Open http://localhost:50505 in your browser")
    print("   2. Try searching for content from your PDF documents")
    print("   3. Check the query history at http://localhost:50505/history")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Semantic Knowledge Base Test Script")
        print("Usage: python test_query_app.py")
        print("\nPrerequisites:")
        print("  1. Run: docker-compose up -d")
        print("  2. Set OPENAI_API_KEY environment variable")
        print("  3. Ensure PDF documents are in ./docs directory")
        sys.exit(0)
    
    test_query_app()
