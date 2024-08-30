import os
from modelscope import snapshot_download
from loguru import logger
import torchaudio
from .cli.cosyvoice import CosyVoice


class CosyVoiceTTS:

    def __init__(self, model_cache_dir="checkpoints/cosyvoice") -> None:
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

        self.model = CosyVoice(f"{model_cache_dir}/CosyVoice-300M-Instruct", load_jit=False)

    def tts(self, text):
        a = self.model.inference_instruct(
            text,
            "中文男",
            "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.",
            stream=False,
        )
        out_f = "out.wav"
        a = next(a)
        print(a)
        torchaudio.save(out_f, a["tts_speech"], 22050)
        return out_f
