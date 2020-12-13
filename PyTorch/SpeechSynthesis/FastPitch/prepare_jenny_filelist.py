import random
import argparse
import json
import time
from pathlib import Path

random.seed(1997)
outputs = {"val": 100, "test": 500, "train": -1}

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    # Mel extraction
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--metadata', required=True,
                        type=str, help='Path to metadata')
    parser.add_argument('--wav-text-filelist', required=True,
                        type=str, help='Path to file with audio paths and text with {} as placeholder')
    return parser


def main():
    parser = argparse.ArgumentParser(description='splits metadata')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    with open(args.metadata,"r") as f:
        lines = f.readlines()

    random.shuffle(lines)

    for out in outputs.keys():
        filename = args.wav_text_filelist.format(out)
        amount = outputs[out] if outputs[out] else len(lines)
        segment = lines[0:amount]
        lines = lines[outputs[out]:]

        # save audio file
        print(filename)
        with open(filename,"w") as o:
            for s in segment:
                p = s.split('|')
                o.write("wavs/{}.wav|{}\n".format(p[0],p[1]))

        # saves mel dur pitch file
        filename = filename.replace("_audio_","_mel_dur_pitch_")
        print(filename)
        with open(filename,"w") as o:
            for s in segment:
                p = s.split('|')
                o.write("mels/{}.pt|durations/{}.pt|pitch_char/{}.pt|{}\n".format(p[0],p[0],p[0],p[1]))

if __name__ == '__main__':
    main()