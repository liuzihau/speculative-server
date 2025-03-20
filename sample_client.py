import random
import asyncio
import httpx
import json
from fastapi import FastAPI, Request
from typing import Dict, Any
import uvicorn

# FastAPI app to simulate the "request server" receiving POST responses
response_app = FastAPI()
received_responses: Dict[str, Any] = {}

@response_app.post("/results")
async def receive_result(request: Request):
    """Endpoint to receive completed task results from DraftModelServer."""
    result = await request.json()
    # draft_id = result["draft_id"]
    # received_responses[draft_id] = result
    with open("./res.json", "w") as f:
        json.dump(result, f, indent=4)
    tokens = [token for token, _ in result["speculative_data"][:10]]
    print(f"Received response: output_tokens[:10]: {tokens}")
    return {"status": "Received"}

async def send_request(data: Dict, prompt: str, response_url: str):
    """Send a generation request to the DraftModelServer with a response URL."""
    data["client_url"] = response_url
    
    # Simulate client request
    max_tokens = random.randint(20, 50)
    temperature = random.randint(0, 1) * 0.1
    data["decoding_setting"]["max_tokens"] = max_tokens
    data["decoding_setting"]["temperature"] = temperature
    data["inputs"]["prompt_text"] = prompt
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{data['server_url']}/generate/", json=data)
        if response.status_code == 200:
            res = response.json()
            print(f"Request sent for {data['server_url']}: Task ID {res['task_id']}")
            return res["task_id"]
        else:
            print(f"Failed to send request for {data['server_url']}: {response.status_code} - {response.text}")
            return None

async def poll_status(server_url: str, task_ids: Dict[str, str]):
    """Poll the server status until all tasks are completed."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{server_url}/status/")
                if response.status_code == 200:
                    status = response.json()
                    ongoing_tasks = status["ongoing_tasks"]
                    print(f"Status check: {ongoing_tasks} tasks ongoing")
                    all_done = all(draft_id in received_responses for draft_id in task_ids.keys())
                    if ongoing_tasks == 0 and all_done:
                        print("All tasks completed and responses received!")
                        break
                else:
                    print(f"Status check failed: {response.status_code} - {response.text}")
            except httpx.RequestError as e:
                print(f"Status check error: {e}")
            await asyncio.sleep(1)

async def main():
    # Configuration
    client_server_url = "http://localhost:8001/results"

    # List of prompts
    draft1_prompts = [
        "The cat sleeps peacefully in the sun",
        # Add more prompts if needed
    ]

    # Start the response server in the background
    config = uvicorn.Config(app=response_app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Send requests sequentially and collect task IDs
    task_ids = {}
    for i, prompt in enumerate(draft1_prompts):
        with open("./config.json", 'r') as f:
            data = json.loads(f.read())
        task_id = await send_request(data=data, prompt=prompt, response_url=client_server_url)
        if task_id:
            task_ids[i] = task_id

    # Wait briefly to ensure server is up, then poll status
    if task_ids:
        print(f"Sent {len(task_ids)} requests. Listening for responses at {client_server_url}...")
        await asyncio.sleep(2)  # Give server time to start
        await poll_status(data["server_url"], task_ids)

    # Shutdown the server gracefully
    server.should_exit = True
    await server_task

if __name__ == "__main__":
    asyncio.run(main())