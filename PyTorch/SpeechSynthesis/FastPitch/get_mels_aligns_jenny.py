import requests
import torch
import numpy as np
import io
from tqdm import tqdm
import os.path
import argparse
import json
import time
from pathlib import Path
from scipy.io.wavfile import write

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    # Mel extraction
    parser.add_argument('-d', '--dataset-path', type=str, default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelist', required=True, type=str, help='Path to file with audio paths and text')
    parser.add_argument("-u", "--url", type=str, default="http://172.17.0.2:5000/tts/generate", help="url for voca tts service")

    return parser


def generate_kwargs_for_voca(text,what):
    headers = {'Authorization': 'Bearer a2f4cd32-f5f6-46fa-952a-b27697382b10',
               'Content-Type': 'application/json; version=1'}
    kwargs = {}
    kwargs['headers'] = headers
    ep = ["weekday_date","speech_acronyms","sms_abbr","month_date","gen_abbr",
    "chat_abbr","cardinal_num","decimal_num","time","gen_date","symbols",
    "year_date","oov_pronounce","units_abbr"]

    json_data = {"sentence": text, "volume": 1, "speed": 1, "sampleRate": 22050, "reply_fields": what,"exclude_preprocess":ep}
    return json_data, kwargs


def getDetailsFromTTS(text,what,url):
    sentence = text
    json_data, kwargs = generate_kwargs_for_voca(sentence, what)
    rawResponse = requests.post(url, json=json_data, stream=True, verify=False, **kwargs)
    reply = torch.load(io.BytesIO(rawResponse.content))
    return reply


def processFile(text,file,folder,url):
    rep = getDetailsFromTTS(text,['mel', 'align_logist', 'align_text','frame_length', 'sample_rate',"signal"],url)
    torch.save(rep['mel'],'./{}/_mels/{}.pt'.format(folder,file))
    torch.save(rep['align_logist'],'./{}/_alignments/{}.pt'.format(folder,file))
    torch.save(rep['align_text'], './{}/_padded_text/{}.pt'.format(folder, file))
    write('./{}/_taco_wav/{}.wav'.format(folder,file), int(rep["sample_rate"]), np.array(rep["signal"].numpy(), np.int16))


def main():
    parser = argparse.ArgumentParser(description='creates Voca mel and alignements for Jenny')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    folder = args.dataset_path
    metadataFile = args.wav_text_filelist
    url = args.url

    with open("{}/{}".format(folder,metadataFile),"r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        completed = False
        while not completed:
            try:
                parts = line.split("|")
                file = parts[0]
                text =  parts[1]
                print(file)
                # if not os.path.exists('./{}/_alignments/{}.pt'.format(folder,file)):
                processFile(text,file,folder,url)
                completed = True
            except Exception as ex:
                print("got exception "+str(ex))
                print(file)


if __name__ == '__main__':
    main()