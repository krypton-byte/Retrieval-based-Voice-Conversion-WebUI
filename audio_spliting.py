from __future__ import unicode_literals
import pip
import os
import sys
import argparse
from pathlib import Path


parse = argparse.ArgumentParser()
parse.add_argument("-n", type=str, help="audio name", required=True)
parse.add_argument("-m", choices=['Splitting', 'Separate'], required=True)
from_ = parse.add_subparsers(title="FROM", dest="dataset", required=True)
yt = from_.add_parser(name="Youtube")
yt.add_argument("-u", type=str, help="url", required=True)
dr = from_.add_parser(name="Drive")
dr.add_argument("-p", type=argparse.FileType("rb"), help="drive path", required=True)
parsed = parse.parse_args()
#@title 1. Install Library for Youtube WAV Download
if parsed.dataset == "Drive":
    print("Dataset is set to Drive. Skipping this section")
elif parsed.dataset == "Youtube":
    if not ('youtubeaudio' in os.listdir('.')):
        os.mkdir('youtubeaudio')
#@title Download Youtube WAV
if parsed.dataset == "Drive":
    print("Dataset is set to Drive. Skipping this section")
elif parsed.dataset == "Youtube":
    try:
        import yt_dlp
        import ffmpeg
    except Exception:
        pip.main(['install', 'yt_dlp', 'ffmpeg'])
        import yt_dlp
        import ffmpeg
    ydl_opts = {
      'format': 'bestaudio/best',
      'postprocessors': [{
          'key': 'FFmpegExtractAudio',
          'preferredcodec': 'wav',
      }],
      "outtmpl": f'youtubeaudio/{parsed.n}',  # this is where you can edit how you'd like the filenames to be formatted
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([parsed.u])

#@title 2. Install Demucs for Separating Audio
try:
    import demucs
except Exception:
    pip.main(['install', '-U', 'demucs'])
#@title Separate Vocal and Instrument/Noise using Demucs
import subprocess
AUDIO_INPUT = f"{Path(__file__).parent.absolute()}/youtubeaudio/{parsed.n}.wav"
command = f"{sys.executable} -m demucs --two-stems=vocals " + (
    parsed.p.name if parsed.dataset == "Drive" else AUDIO_INPUT
)
result = subprocess.run(command.split(), stdout=subprocess.PIPE)
print(result.stdout.decode())
if not Path(f'/content/drive/MyDrive/audio/{parsed.n}').exists():
    os.makedirs(f'/content/drive/MyDrive/audio/{parsed.n}')
os.system(f'cp -r {Path(__file__).parent.absolute()}/separated/htdemucs/{parsed.n}/* /content/drive/MyDrive/audio/{parsed.n}')
if parsed.dataset == "Youtube":
    os.system(f'cp -r {Path(__file__).parent.absolute()}/youtubeaudio/{parsed.n}.wav /content/drive/MyDrive/audio/{parsed.n}')
#@title 3. Split The Audio into Smaller Duration Before Training
if parsed.m == "Separate":
    print("Mode is set to Separate. Skipping this section")
elif parsed.m ==  "Splitting":
    pip.main(['install', 'numpy', 'librosa', 'soundfile'])
    if not Path(f'dataset/{parsed.n}').exists():
        os.makedirs(f'dataset/{parsed.n}')
#@title
try:
    import numpy as np
    import librosa 
    import soundfile
except Exception:
    pip.main(['install', 'numpy', 'librosa', 'soundfile'])
    import numpy as np
    import librosa 
    import soundfile

# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

if parsed.m == "Separate":
    print("Mode is set to Separate. Skipping this section")
elif parsed.m ==  "Splitting":
    audio, sr = librosa.load(f'{Path(__file__).parent.absolute()}/separated/htdemucs/{parsed.n}/vocals.wav', sr=None, mono=False)  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=200,
        hop_size=10,
        max_sil_kept=500
    )
    chunks = slicer.slice(audio)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        soundfile.write(f'{Path(__file__).parent.absolute()}/dataset/{parsed.n}/split_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.

if parsed.m == "Separate":
    print("Mode is set to Separate. Skipping this section")
elif parsed.m == "Splitting":
    if not Path(f'{Path(__file__).parent.absolute()}/dataset/{parsed.n}').exists():
        os.makedirs(f'{Path(__file__).parent.absolute()}/dataset/{parsed.n}')
    if not Path(f'/content/drive/MyDrive/dataset/{parsed.n}').exists():
        os.makedirs(f'/content/drive/MyDrive/dataset/{parsed.n}/')
    os.system(f'cp -r {Path(__file__).parent.absolute()}/dataset/{parsed.n}/* /content/drive/MyDrive/dataset/{parsed.n}/')
