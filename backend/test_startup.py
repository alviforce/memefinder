import asyncio
import sys
import os

from main import app, lifespan

async def test_startup():
    print("Testing backend startup...", flush=True)
    try:
        async with lifespan(app):
            print("Lifespan started successfully!", flush=True)
    except Exception as e:
        print(f"Startup error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(test_startup())
