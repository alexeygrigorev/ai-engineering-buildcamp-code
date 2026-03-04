import asyncio
import time
from dotenv import load_dotenv
from logs.sql import SQLiteStorage
from tools import create_documentation_tools_cached
from logs.judge.pipeline import EvaluationPipeline

load_dotenv()

async def main():
    print("Starting Evaluation Pipeline...")
    
    storage = SQLiteStorage()
    search_tools = create_documentation_tools_cached()
    pipeline = EvaluationPipeline(storage, search_tools)
    
    while True:
        try:
            await pipeline.run_once()
        except Exception as e:
            print(f"Error in evaluation loop: {e}")
        
        # Poll every 30 seconds
        print("Waiting for new logs...")
        await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Evaluation Pipeline stopped by user.")
