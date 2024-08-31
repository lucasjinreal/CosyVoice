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
import io
from typing import Literal, TypedDict, Union
import fastapi
from fastapi import FastAPI, Request, Response
import librosa
import numpy as np
from uvicorn.config import Config
from uvicorn import Server
from .api import CosyVoiceTTS
import torch
from pydantic import BaseModel
import os
import logging
from scipy.io.wavfile import write

# logging.disable(logging.INFO)
logger_librosa = logging.getLogger("librosa")
logger_librosa.setLevel(logging.CRITICAL)
logging.getLogger("numba").setLevel(logging.WARNING)


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

    res = model.tts_instruct(txt, request.voice, return_format="wav")
    # print(res_f)
    res = next(res)
    res = res.cpu().numpy()[0]
    # print(res, res.dtype)
    response_format = request.response_format
    # with open(res_f, "rb") as file:
    #     buffer = file.read()

    # return is sr 16k int16 wav data, it can be directly played
    audio_chunk = librosa.resample(res, orig_sr=24000, target_sr=16000)
    audio_int16 = (audio_chunk * 32768).astype(np.int16)
    buffer = io.BytesIO()
    write(buffer, 16000, audio_int16)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type=f"audio/{response_format}")


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
