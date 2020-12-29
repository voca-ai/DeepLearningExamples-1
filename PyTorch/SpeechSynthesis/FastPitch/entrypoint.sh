#!/bin/sh

/bin/bash /workspace/fastpitch/scripts/download_waveglow.sh
/bin/bash /workspace/fastpitch/scripts/download_fastpitch.sh
python /workspace/fastpitch/flask_app.py
