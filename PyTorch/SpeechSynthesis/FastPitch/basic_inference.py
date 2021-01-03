# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import io
import torch
import warnings
import argparse
import re
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write
from waveglow.denoiser import Denoiser
from inference import load_and_setup_model, prepare_input_sequence, MeasureTime, parse_args, build_pitch_transformation
from common.text.symbols import get_symbols


# TODO  take frame_size out to configuration
frame_length = 256 / 22050


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
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
    fields = {"text": [args.input]}
    batches = prepare_input_sequence(
        fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None))
    gen_measures = MeasureTime()
    waveglow_measures = MeasureTime()
    gen_kw = {'pace': args.pace,
              'speaker': args.speaker,
              'pitch_tgt': None,
              'pitch_transform': build_pitch_transformation(args)}
    all_utterances = 0
    all_samples = 0
    all_letters = 0
    all_frames = 0
    # reps = args.repeats
    # log_enabled = reps == 1
    # for rep in (tqdm.tqdm(range(reps)) if reps > 1 else range(reps)):
    # TODO right now code can only run with 1 batch, should implement method for many batches
    for b in batches:
        if generator is None:
            mel, mel_lens = b['mel'], b['mel_lens']
        else:
            with torch.no_grad(), gen_measures:
                mel, mel_lens, dur_pred, pitch_pred = generator(
                    b['text'], b['text_lens'], **gen_kw)

            letters = get_symbols()
            text = "".join([letters[i] for i in b['text'][0].tolist()])
            text = text[0:b['text_lens'][0]]
            text = text.strip()
            spaces = [m.start() for m in re.finditer(' ', text)]
            timings = np.cumsum(np.round(dur_pred.cpu()))
            word_timings = []
            for idx in range(len(spaces)):
                if idx == 0:
                    start = timings[0]
                    end = timings[spaces[idx] - 1]
                elif idx < len(spaces) - 1:
                    start = timings[spaces[idx - 1] + 1]
                    end = timings[spaces[idx] - 1]
                else:
                    start = timings[spaces[idx] + 1]
                    end = timings[-1]
                word_timings.append([start, end])
            print(word_timings)

            # word_starts = np.concatenate([np.array([0]), timings[np.array(spaces) + 1]])
            # print(word_starts)

            all_letters += b['text_lens'].sum().item()
            all_frames += mel.size(0) * mel.size(2)
        if waveglow is not None:
            with torch.no_grad(), waveglow_measures:
                audios = waveglow(mel, sigma=args.sigma_infer)
                audios = denoiser(audios.float(),
                                  strength=args.denoising_strength
                                  ).squeeze(1)
            all_utterances += len(audios)
            all_samples += sum(audio.size(0) for audio in audios)
            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * args.stft_hop_length]
                if args.fade_out:
                    fade_len = args.fade_out * args.stft_hop_length
                    fade_w = torch.linspace(1.0, 0.0, fade_len)
                    audio[-fade_len:] *= fade_w.to(audio.device)
                audio = audio / torch.max(torch.abs(audio))
                with io.BytesIO() as f:
                    write(f, args.sampling_rate, audio.cpu().numpy())
                    f.seek(0)
                result = {
                    "wav": f.read(),
                    "frame_length": frame_length,
                    "word_timings": word_timings
                }
                buffer = io.BytesIO()
                torch.save(result, buffer)
                buffer.seek(0)
                return buffer


if __name__ == '__main__':
    main()
