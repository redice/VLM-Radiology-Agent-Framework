import argparse
import logging
import tempfile
import os
import time
from typing import List

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from generator import ChatHistory, M3Generator, SessionVariables, new_session_variables
from pydantic import HttpUrl

load_dotenv()

logger = logging.getLogger(__name__)
currentDir = os.getcwd()
relativeDir = os.getenv("RELATIVE_DIRECTORY")

global generator

# FastAPI app setup
app = FastAPI(
    title="VILA-M3",
    openapi_url="/openapi.json",
    docs_url=None,
    redoc_url="/docs",
)

infoRouter = APIRouter(
    prefix="/info",
    tags=["App Info"],
    responses={404: {"description": "Not found"}},
)


@infoRouter.get("/", summary="Get App Info")
async def api_app_info():
    """Get model info."""
    return "VILA-M3 model"


app.include_router(infoRouter)


# Replace with your actual VLM/M3 Gradio interface setup
def create_demo(source, model_path, conv_mode):
    generator = M3Generator(source=source, model_path=model_path, conv_mode=conv_mode)

    return generator

@app.post("/single")
async def execute_api(image_file: UploadFile = File(...), prompt_text: str = Form(...)):
    try:
        sv = SessionVariables()
        chat_history = ChatHistory()

        temp_image_path = os.path.join(currentDir, relativeDir, image_file.filename)
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await image_file.read())

        logger.debug(f"Received images: {temp_image_path}, prompt: {prompt_text}")

        sv.image_url = temp_image_path
        sv.slice_index = 57  # Example slice index (adjust as needed) - This should come from Slicer!!!

        sv, chat_history = generator.process_prompt(prompt_text, sv, chat_history)

        response_message = chat_history.messages[-1] if chat_history.messages else "No response generated"

        response = {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "vila-m3",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": response_message}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Clean up temporary file
        os.remove(temp_image_path)

        return JSONResponse(content=response)

    except FileNotFoundError:
        logger.error("Image file not found.")
        raise HTTPException(status_code=404, detail="Image file not found")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_api(image_files: List[UploadFile] = File(...), prompt_text: str = Form(...)):
    try:
        sv = SessionVariables()
        chat_history = ChatHistory()

        file_paths = []
        for image_file in image_files:
            temp_image_path = os.path.join(currentDir, relativeDir, image_file.filename)
            with open(temp_image_path, "wb") as buffer:
                buffer.write(await image_file.read())
            file_paths.append(temp_image_path)
        logger.debug(f"Received images: {file_paths}, prompt: {prompt_text}")

        sv.image_url = file_paths
        sv.slice_index = 57  # Example slice index (adjust as needed) - This should come from Slicer!!!

        sv, chat_history = generator.process_prompt(prompt_text, sv, chat_history)

        response_message = chat_history.messages[-1] if chat_history.messages else "No response generated"

        response = {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "vila-m3",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": response_message}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Clean up temporary file
        for image_file in image_files:
            temp_image_path = os.path.join(currentDir, relativeDir, image_file.filename)
            os.remove(temp_image_path)

        return JSONResponse(content=response)

    except FileNotFoundError:
        logger.error("Image file not found.")
        raise HTTPException(status_code=404, detail="Image file not found")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom swagger UI."""
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace("showExtensions: true,", "showExtensions: true, defaultModelsExpandDepth: -1,")
    return HTMLResponse(body)


if __name__ == "__main__":
    logfile = os.getenv("LOGFILE")
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in " "function %(funcName)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    # TODO: Add the argument to load multiple models from a JSON file
    parser.add_argument(
        "--convmode",
        type=str,
        default="llama_3",
        help="The conversation mode to use. For 8B models, use 'llama_3'. For 3B and 13B models, use 'vicuna_v1'.",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="MONAI/Llama3-VILA-M3-8B",
        help=(
            "The path to the model to load. "
            "If source is 'local', it can be '/data/checkpoints/vila-m3-8b'. If "
            "If source is 'huggingface', it can be 'MONAI/Llama3-VILA-M3-8B'."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.getenv("PORT"),
        help="The port to run the Gradio server on.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        help="The source of the model. Option is 'huggingface' or 'local'.",
    )
    args = parser.parse_args()
    generator = create_demo(args.source, args.modelpath, args.convmode)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
