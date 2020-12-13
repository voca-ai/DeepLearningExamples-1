# adds speaker
import argparse
import json
import time
from pathlib import Path
import random

import parselmouth
import torch
import numpy as np



def parse_args(parser):
    """
    Parse commandline arguments.
    """
    #parser.add_argument('--wav-text-filelist', required=True, type=str, help='Path to file with audio paths and text')
    return parser

lables = [    ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B',
              'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b',
              'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
              'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' ,' ']
maping = {}
demapping = {}
cc = 1
for c in lables:
    maping[c] = cc
    demapping[cc] = c
    cc+=1


def updated_text(text_file,original_text):
    tf = torch.load(text_file)
    ptxt = "".join([demapping[int(t)] for t in list(tf)])+"\n"
    text = " " + original_text.replace('\n','') + "    \n"
    if len(text)!=len(ptxt):
        # we will be loosing caps, but gaining low level pronounciation
        return ptxt
    else:
        # nothing to loose or win.
        return text



def main():
    parser = argparse.ArgumentParser(description='creates Voca mel and alignements for Jenny')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    users = {"jenny":"Jenny", "ljs":"LJSpeech-1.1"}
    files = ['train', 'test', 'val']

# original code works with misalignment
#     for file in files:
#         out_lines = []
#         for u,user in enumerate(users.keys()):
#             filename = "filelists/{}_mel_dur_pitch_text_{}_filelist.txt".format(user,file)
#             with open(filename,"r") as f:
#                 lines = f.readlines()
#             for line in lines:
#                 l = line.replace("mels/","{}/mels/".format(users[user])) \
#                     .replace("durations/","{}/durations/".format(users[user])) \
#                     .replace("pitch_char/","{}/pitch_char/".format(users[user])) \
#                     .replace("\n", "|{}\n".format(u))
#                 out_lines.append(l)
#         random.shuffle(out_lines)
#
#         filename = "filelists/all_mel_dur_pitch_text_{}_filelist.txt".format(file)
#         with open(filename, "w") as f:
#             for l in out_lines:
#                 f.write(l)

#
# updated text tries to fix fix misalignment using hte padded text
#
    for file in files:
        out_lines = []
        for u,user in enumerate(users.keys()):
            filename = "filelists/{}_mel_dur_pitch_text_{}_filelist.txt".format(user,file)
            with open(filename,"r") as f:
                lines = f.readlines()
                for line in lines:

                    if user=="jenny":
                        # fix text
                        s = line.split('|')
                        text_file = users[user]+"/"+s[0].replace("mels","_padded_text")
                        s[-1] = updated_text(text_file,s[-1])
                        line = "|".join(s)
                        line = line.replace("mels/","{}/_mels/".format(users[user]))
                    else:
                        line = line.replace("mels/", "{}/mels/".format(users[user]))
                    # in any case....

                    l = line.replace("durations/","{}/durations/".format(users[user])) \
                        .replace("pitch_char/","{}/pitch_char/".format(users[user])) \
                        .replace("\n", "|{}\n".format(u))
                    out_lines.append(l)
        random.shuffle(out_lines)

        filename = "filelists/all_mel_dur_pitch_text_{}_filelist.txt".format(file)
        with open(filename, "w") as f:
            for l in out_lines:
                f.write(l)



if __name__ == '__main__':
    main()
