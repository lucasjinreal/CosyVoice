import os
from modelscope import snapshot_download
from loguru import logger
import torchaudio
from .cli.cosyvoice import CosyVoice
import torch


class CosyVoiceTTS:

    def __init__(self, model_cache_dir="checkpoints/cosyvoice", device="cuda") -> None:
        if not os.path.exists(model_cache_dir):
            logger.info(f"downloading cosyvoice from modelscope.")
            snapshot_download(
                "iic/CosyVoice-300M", local_dir=f"{model_cache_dir}/CosyVoice-300M"
            )
            snapshot_download(
                "iic/CosyVoice-300M-SFT",
                local_dir=f"{model_cache_dir}/CosyVoice-300M-SFT",
            )
            snapshot_download(
                "iic/CosyVoice-300M-Instruct",
                local_dir=f"{model_cache_dir}/CosyVoice-300M-Instruct",
            )
            snapshot_download(
                "iic/CosyVoice-ttsfrd", local_dir=f"{model_cache_dir}/CosyVoice-ttsfrd"
            )
            logger.info("cosyvoice model downloaded.")

        self.model = CosyVoice(
            f"{model_cache_dir}/CosyVoice-300M-Instruct", load_jit=False, device=device
        )

    def tts_instruct(
        self,
        text,
        spk_id="中文男",
        prompt="Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.",
        return_format="wav",
        stream=False,
    ):
        a = self.model.inference_instruct(
            text,
            # "中文男",
            spk_id,
            prompt,
            stream=stream,
        )

        if return_format == "wav":
            if stream:
                for itm in a:
                    yield itm["tts_speech"]
            else:
                a = next(a)
                yield a["tts_speech"]
        elif return_format == "file" and not stream:
            out_f = "out.wav"
            a = next(a)
            torchaudio.save(out_f, a["tts_speech"], 22050)
            yield out_f
        else:
            ValueError(
                f"unsupported combination of stream and return_format: {stream} {return_format}"
            )

    def tts_test(self, text):
        a = self.model.inference_instruct(
            text,
            # "中文男",
            "日语男",
            "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.",
            stream=False,
        )
        out_f = "out.wav"
        a = next(a)
        print(a)
        torchaudio.save(out_f, a["tts_speech"], 22050)
        return out_f
