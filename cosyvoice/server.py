"""
A simple entrace serving cosyvoice as TTS service

like:

cosyvoice -h
cosyvoie

then you will have a TTS service running on GPU by default, default model is instruct
you can only server one model at once.
"""

import argparse
from contextlib import asynccontextmanager
from typing import Literal, TypedDict, Union
import fastapi
from fastapi import FastAPI, Request, Response
from uvicorn.config import Config
from uvicorn import Server
from .api import CosyVoiceTTS
import torch
from pydantic import BaseModel
import os
import logging


# logging.disable(logging.CRITICAL)


class SpeechCreateParams(BaseModel):
    input: str
    model: Union[str, Literal["tts-1", "tts-1-hd"]] = "tts-1"
    voice: str = "中文男"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = 1.0


async def startup():
    print("started up..")


async def shutdown():
    print("shutting down..")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/audio/speech", response_model=SpeechCreateParams)
async def text_to_speech(request: SpeechCreateParams, request_raw: Request):
    txt = request.input
    # do tts how to return?
    global model

    res_f = model.tts_instruct(txt, request.voice, return_format="file")
    # print(res_f)
    res_f = next(res_f)
    response_format = request.response_format
    with open(res_f, "rb") as file:
        buffer = file.read()
    return Response(content=buffer, media_type=f"audio/{response_format}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default=8088)
    parser.add_argument("--type", type=str, default="instruct")
    parser.add_argument("--https", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CosyVoiceTTS(
        device=device,
        model_cache_dir=os.path.expanduser("~/cosyvoice_models/cosyvoice"),
    )

    import asyncio

    http_config = Config(app=app, host=args.ip, port=args.port, log_level="info")
    http_server = Server(config=http_config)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(http_server.serve()))
