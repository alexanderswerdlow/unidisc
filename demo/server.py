import asyncio
import base64
import logging
import multiprocessing as mp
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Union
import random
import json
import hydra
import torch
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uvicorn import run

from decoupled_utils import breakpoint_on_error
from demo.api_data_defs import ChatMessage, ChatRequest, ContentPart
from demo.inference_utils import (convert_request_base64_to_pil,
                                  convert_request_pil_to_base64,
                                  trim_merge_messages)
from utils import set_omega_conf_resolvers

logger = logging.getLogger("uvicorn.error")

mp.set_start_method('spawn', force=True)

set_omega_conf_resolvers()


async def dummy_response(messages: List[Dict[str, Any]]) -> ChatRequest:
    await asyncio.sleep(0.1)
    response_content = []
    for msg in messages:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "text":
                    response_content.append(ContentPart(type="text", text="Response to: " + item["text"]))
                elif item["type"] == "image_url":
                    response_content.append(ContentPart(type="text", text="Image received and processed."))
    
    image_path = Path("static/0457_01.jpg")  # Replace with a real image path
    if image_path.is_file():
        with image_path.open("rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            response_content.append(ContentPart(
                type="image_url",
                image_url={"url": f"data:image/jpeg;base64,{base64_str}"}
            ))
    else:
        logger.warning(f"Image file not found at {image_path}")
    
    return ChatRequest(messages=[ChatMessage(role="assistant", content=response_content)])

def call_model(messages: List[Dict[str, Any]], inference) -> ChatRequest:
    print(f"input messages: {messages}")
    returned_messages = inference(messages)
    openai_messages = convert_request_pil_to_base64(returned_messages)
    return openai_messages

def generate_response(messages: List[Dict[str, Any]], inference, dummy_response: bool = False) -> ChatRequest:
    if dummy_response:
        return dummy_response(messages)
    else:
        return call_model(messages, inference)

def call(inference, request: ChatRequest):
    try:
        print(f"Hash: {request.request_hash}")
        output_dir = Path(f"{Path(__file__).parent}/outputs/responses")
        filename = output_dir / f"{request.request_hash}.json"

        if request.request_hash is not None and filename.exists():
            with open(filename, "r") as f:
                generated_json = json.load(f)
                print(f"Response loaded from {filename}")
        else:
            processed_messages = convert_request_base64_to_pil(request)
            processed_messages = trim_merge_messages(processed_messages)
            generated: ChatRequest = generate_response(processed_messages, inference)
            generated_json = generated.messages[-1].model_dump()


        if request.request_hash is not None and not filename.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w") as f:
                json.dump(generated.messages[-1].model_dump(), f, indent=2)
            
            print(f"Response saved to {filename}")

        # OpenAI format
        return JSONResponse({
            "id": "cmpl-000",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "choices": [{
                "index": 0,
                "message": generated_json,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        from traceback import format_exc
        logger.error(f"Error processing request: {str(e)}")
        logger.error(format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def gpu_worker(gpu_id, config, request_queue, response_queue):
    torch.cuda.set_device(gpu_id) # We use this instead of CUDA_VISIBLE_DEVICES since the user may have set.
    from demo.inference import setup
    inference = setup(config)
    print(f"GPU {gpu_id} Initialized inference")
    while True:
        # Wait for a new request (blocking call)
        print(f"GPU {gpu_id} Waiting for request")
        request_data = request_queue.get()
        print(f"GPU {gpu_id} Received request")
        if request_data is None:
            print(f"GPU {gpu_id} Received shutdown signal")
            break  # a way to shut down this worker gracefully
        try:
            # Process the request – note that this call is synchronous
            print(f"GPU {gpu_id} Processing request")
            start_time = time.time()
            result = call(inference, request_data)
            print(f"GPU {gpu_id} Finished processing request in {time.time() - start_time} seconds")
            response_queue.put(result)
            print(f"GPU {gpu_id} Put result in response queue")
        except Exception as e:
            print(f"GPU {gpu_id} Error processing request {request_data}: {e}")
            response_queue.put(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if we're in development mode
    dev_mode = getattr(app.config, "dev_mode", False)
    app.state.dev_mode = dev_mode
    print(f"Dev mode: {dev_mode}")
    
    if dev_mode:
        # Development mode: Single synchronous GPU process
        logging.info("Starting in DEVELOPMENT mode - synchronous operation, no multiprocessing")
        from demo.inference import setup
        app.state.inference = setup(app.config)
        yield
    else:
        # Normal mode with worker processes
        app.state.worker_lock = asyncio.Lock()
        workers = []
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of GPUs: {num_gpus}")
        for gpu_id in range(num_gpus):
            req_q = mp.Queue(maxsize=1)  # enforce one request at a time
            res_q = mp.Queue()
            p = mp.Process(target=gpu_worker, args=(gpu_id, app.config, req_q, res_q))
            p.start()
            workers.append({"process": p, "req_q": req_q, "res_q": res_q})
            logging.info(f"Started worker {gpu_id}")
        
        app.state.workers = workers
        yield
        # On shutdown: signal all workers to stop and join them
        for worker in app.state.workers:
            worker["req_q"].put(None)
        for worker in app.state.workers:
            worker["process"].join()
            logger.info("Worker process joined.")


app = FastAPI(title="Multimodal VLM Endpoint", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.workers = []
logger = logging.getLogger("uvicorn")


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    if getattr(app.state, "dev_mode", False):
        return call(app.state.inference, request)
    
    worker = None
    async with app.state.worker_lock:
        while worker is None:
            # Shuffle workers each time to distribute load
            workers = list(enumerate(app.state.workers))
            random.shuffle(workers)
            for i, w in workers:
                print(f"Trying to assign request to worker {i}")
                try:
                    w["req_q"].put_nowait(request)
                    worker = w
                    print(f"Assigned request to worker {w['process'].name}")
                    break
                except mp.queues.Full:
                    print(f"Worker {w['process'].name} is full")
                    continue
            if worker is None:
                await asyncio.sleep(0.1)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, worker["res_q"].get)
    if isinstance(result, Exception):
        raise HTTPException(status_code=500, detail=str(result))
    return result

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error("Request body: %s", body)
    logger.error("Validation errors: %s", exc.errors())
    logger.error("Original body: %s", exc.body)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()
def main(config):
    with breakpoint_on_error():
        app.config = config
        dev_mode = getattr(config, "dev_mode", False)
        app.state.dev_mode = dev_mode
        run(app, host="0.0.0.0", port=getattr(config, "port", 8001))
    
if __name__ == "__main__":
    main()