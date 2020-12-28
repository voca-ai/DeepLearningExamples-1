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
# import tqdm
from pathlib import Path
from scipy.io.wavfile import write
from waveglow.denoiser import Denoiser
from inference import load_and_setup_model, prepare_input_sequence, MeasureTime, parse_args, build_pitch_transformation


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
    for b in batches:
        if generator is None:
            mel, mel_lens = b['mel'], b['mel_lens']
        else:
            with torch.no_grad(), gen_measures:
                mel, mel_lens, *_ = generator(
                    b['text'], b['text_lens'], **gen_kw)
            gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
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
            # waveglow_infer_perf = (
            #     audios.size(0) * audios.size(1) / waveglow_measures[-1])
            # if args.output is not None and reps == 1:
            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * args.stft_hop_length]
                if args.fade_out:
                    fade_len = args.fade_out * args.stft_hop_length
                    fade_w = torch.linspace(1.0, 0.0, fade_len)
                    audio[-fade_len:] *= fade_w.to(audio.device)
                audio = audio / torch.max(torch.abs(audio))
                # fname = b['output'][i] if 'output' in b else f'audio_{i}.wav'
                # audio_path = Path(args.output, fname)
                # torch.save(torch.squeeze(mel, 0), str(audio_path).replace('.wav','_mel.pt'),
                #            _use_new_zipfile_serialization=False)
                with io.BytesIO() as f:
                    write(f, args.sampling_rate, audio.cpu().numpy())
                    f.seek(0)
                    return f.read()


if __name__ == '__main__':
    main()
