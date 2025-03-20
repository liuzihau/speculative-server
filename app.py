import asyncio
from fastapi import FastAPI, Body
from pydantic import BaseModel
from celery import Celery
from typing import Dict, List, Any, Union
import httpx
import json
from pyngrok import ngrok
import uvicorn

from model import HFModelEngine

# Config
celery_server_name = "draft_model_server"
draft_id = 1
broker_url = "redis://localhost:6379/0"
backend_url = "redis://localhost:6379/0"

server_type = "draft"
model_name = "gpt2"
device = "cpu"

# server_type = "verify"
# model_name = "gpt2-large"
# device = "cuda:0"

# Initialize Celery
celery_app = Celery(
    celery_server_name,
    broker=broker_url,
    backend=backend_url
)

# Initialize FastAPI
app = FastAPI()
engine = HFModelEngine(model_name=model_name, batch_size=4, device=device)

class RequestBody(BaseModel):
    server_name: str
    server_url: str
    verify_url: str
    decoding_setting: Dict
    inputs: Dict
    outputs: List
    draft_server_setting: Dict
    speculative_data: List
    is_finished: bool
    results: Dict

# Celery tasks
@celery_app.task(bind=True, name="process_draft_request")
def process_draft_request(self, payload):
    async def draft_generate():
        if payload["is_finished"]:
            print("is_finished")
            async with httpx.AsyncClient() as client:
                await client.post(payload["client_url"], json=payload)
            return payload
        
        if payload["inputs"]["input_ids"] is not None:
            prompt = payload["inputs"]["input_ids"]
        else:
            prompt = payload["inputs"]["prompt_text"]
        output = engine.draft_generate(prompt=prompt, 
                                       max_tokens=payload["decoding_setting"]["max_tokens"], 
                                       proposal_length=payload["decoding_setting"]["proposal_length"],
                                       temperature=payload["decoding_setting"]["temperature"]
                                       )
        payload["inputs"]["input_ids"] = output["input_ids"]
        payload["speculative_data"] = output["guesses"]
        
        async with httpx.AsyncClient() as client:
            await client.post(f"{payload['verify_url']}/verify/", json=payload)
        return payload

    if payload["draft_server_setting"]["draft_id"] is None:
        payload["draft_server_setting"]["draft_id"] = draft_id

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(draft_generate())

@celery_app.task(bind=True, name="process_verify_request")
def process_verify_request(self, payload):
    async def verify_proposal_token():
        (new_input_ids, 
         output_results, 
         accepted_tokens, 
         gen_tokens, 
         is_finished) = engine.verify_generate(
            input_ids=payload["inputs"]["input_ids"], 
            draft_guesses=payload["speculative_data"], 
            temperature=payload["decoding_setting"]["temperature"], 
            max_tokens=payload["decoding_setting"]["max_tokens"],
            gen_tokens=payload["decoding_setting"]["gen_tokens"]
            )
        payload["is_finished"] = is_finished
        payload["inputs"]["input_ids"] = new_input_ids
        payload["decoding_setting"]["gen_tokens"] = gen_tokens
        payload["results"]["text"] = output_results
        payload["outputs"].append({"accepted_tokens": accepted_tokens})
        print(f"is_finished: {is_finished}")
        async with httpx.AsyncClient() as client:
            await client.post(f"{payload['server_url']}/generate/", json=payload)
        return payload
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(verify_proposal_token())

# FastAPI endpoints
@app.post("/generate/")
async def generate_text(request: RequestBody = Body(...)):
    task = celery_app.send_task("process_draft_request", args=[request.model_dump()])
    return {"task_id": task.id, "status": "Processing", "task type": "draft"}

@app.post("/verify/")
async def verify_text(request: RequestBody = Body(...)):
    task = celery_app.send_task("process_verify_request", args=[request.model_dump()])
    return {"task_id": task.id, "status": "Processing", "task type": "verify"}

@app.get("/status/")
async def get_status():
    from celery.app.control import Inspect
    i = Inspect(app=celery_app)
    active_tasks = i.active()
    if active_tasks:
        num_active = sum(len(tasks) for tasks in active_tasks.values())
    else:
        num_active = 0
    return {"ongoing_tasks": num_active}

# Start server with ngrok
def start_server_with_ngrok():
    local_port = 8000
    # ngrok.set_auth_token("YOUR_AUTH_TOKEN_HERE")  # Uncomment and add your token if needed
    public_url = ngrok.connect(local_port, "http").public_url
    print(f"Server is publicly accessible at: {public_url}")
    config = uvicorn.Config(app=app, host="0.0.0.0", port=local_port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

if __name__ == "__main__":
    start_server_with_ngrok()