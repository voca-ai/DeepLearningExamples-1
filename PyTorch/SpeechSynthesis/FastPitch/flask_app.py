import sys
from flask import Flask, Response, request
import json
import argparse
import torch
import warnings
import logging
from waveglow.denoiser import Denoiser
from inference import load_and_setup_model, parse_args

from basic_inference import main

app = Flask(__name__)

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)

arguments = ["--cuda", "--fastpitch", "pretrained_models/fastpitch/nvidia_fastpitch_200518.pt",
                 "--waveglow", "pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt",
                 "--wn-channels", "256", "-o", "output/wavs_devset10", "--pitch-transform-custom", "-i", "text"]
sys.argv += arguments
parser = argparse.ArgumentParser(description='Flask engine args', allow_abbrev=False)
parser = parse_args(parser)
args, unk_args = parser.parse_known_args()

logger.info("loading models")
device = torch.device('cuda' if args.cuda else 'cpu')
generator = load_and_setup_model(
    'FastPitch', parser, args.fastpitch, args.amp, device,
    unk_args=unk_args, forward_is_infer=True, ema=args.ema,
    jitable=args.torchscript)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waveglow = load_and_setup_model(
        'WaveGlow', parser, args.waveglow, args.amp, device,
        unk_args=unk_args, forward_is_infer=True, ema=args.ema)
denoiser = Denoiser(waveglow).to(device)
waveglow = getattr(waveglow, 'infer', waveglow)
logger.info("finished loading models")


@app.route("/generate", methods=["POST"])
def generate_audio():
    data = json.loads(request.get_data())
    text = data.get("text")
    logger.info("request received", extra={"text": text})
    sys.argv[-1] = text
    return Response(main(generator, denoiser, waveglow, device), status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
