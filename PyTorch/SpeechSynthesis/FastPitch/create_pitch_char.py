import numpy as np
import argparse
import json
import time
import torch
from pathlib import Path
from common import utils

def normalize_pitch_vectors(pitch_vecs):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for v in pitch_vecs])
    mean, std = np.mean(nonzeros), np.std(nonzeros)
    for v in pitch_vecs:
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0

    return mean, std


def save_stats(dataset_path, wav_text_filelist, feature_name, mean, std):
    fpath = utils.stats_filename(dataset_path, wav_text_filelist, feature_name)
    print("saving {}".format(fpath))
    with open(fpath, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=4)


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str, default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelist', required=True, type=str, help='Path to file with audio paths and text')
    return parser


def main():
    parser = argparse.ArgumentParser(description='PyTorch TTS Data Pre-processing')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    filename = args.wav_text_filelist #"Jenny/metadata2.csv"
    folder = args.dataset_path #'Jenny'

    with open(filename,"r") as f:
        lines = f.readlines()

    print("processing {} lines".format(len(lines)))
    pitch_vecs_list = []
    for line in lines:
        fname = line.split("|")[0]
        fpath = fname.replace("wavs","pitch_char").replace(".wav",".pt")
        pitch = torch.load(fpath)
        pitch_vecs_list.append(pitch)

    mean, std = normalize_pitch_vectors(pitch_vecs_list)
    print("found mean {} std {}".format(mean,std))
    save_stats(folder, filename, 'pitch_char', mean, std)

if __name__ == '__main__':
    main()
