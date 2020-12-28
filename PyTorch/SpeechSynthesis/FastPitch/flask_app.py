import sys
from flask import Flask, Response, request
import json

from basic_inference import main

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_audio():
    data = json.loads(request.get_data())
    text = data.get("text")
    arguments = ["--cuda", "--fastpitch", "pretrained_models/fastpitch/nvidia_fastpitch_200518.pt",
                 "--waveglow", "pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt",
                 "--wn-channels", "256", "-o", "output/wavs_devset10", "-i", text]
    sys.argv += arguments
    return Response(main(), status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
