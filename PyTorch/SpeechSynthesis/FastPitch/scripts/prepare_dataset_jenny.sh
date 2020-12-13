#!/usr/bin/env bash

set -e

TACOTRON2="pretrained_models/tacotron2/nvidia_tacotron2pyt_fp16.pt"

#python3 ./get_mels_aligns_jenny.py -d Jenny -wav-text-filelist metadata.csv -u "http://172.17.0.2:5000/tts/generate"
#python3 ./prepare_jenny_filelist.py -d Jenny --metadata Jenny/metadata.csv --wav-text-filelist './filelists/jenny_audio_text_{}_filelist.txt'

for FILELIST in jenny_audio_text_train_filelist.txt \
                jenny_audio_text_val_filelist.txt \
                jenny_audio_text_test_filelist.txt \
; do
    python extract_mels_jenny.py \
        --cuda \
        --dataset-path Jenny \
        --wav-text-filelist filelists/${FILELIST} \
        --batch-size 256 \
        --extract-mels \
        --extract-durations \
        --extract-pitch-char \
        --pitch-mean 220.9216 \
        --pitch-std 64.7292 \
        --tacotron2-checkpoint ${TACOTRON2}
done

#
# after LJS is prepared - can run a merge
#
python3 ./merge_datasets.py