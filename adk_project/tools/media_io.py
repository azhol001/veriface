# adk_project/tools/media_io.py
# Minimal media I/O for VeriFace
# - extract frames at fixed FPS
# - load mono audio at a fixed sample rate
# - quick CLI to sanity-check a file

from __future__ import annotations
import argparse
from typing import Tuple, List

import numpy as np
import librosa
from moviepy.editor import VideoFileClip


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """
    Returns:
      y: mono waveform (float32, range ~[-1, 1])
      sr: sample rate (int)
      dur_s: duration in seconds (float)
    """
    # First try: librosa direct
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        dur_s = float(len(y) / sr)
        return y.astype(np.float32), sr, dur_s
    except Exception:
        pass  # fall through to robust temp-wav route

    # Robust route: write a temporary WAV via MoviePy, then read with librosa
    import tempfile, os
    clip = VideoFileClip(path)
    if clip.audio is None:
        clip.close()
        raise RuntimeError("No audio track found in file.")

    dur_s = float(clip.duration)
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tmp_audio.wav")
        # Write PCM WAV at target_sr; this uses bundled ffmpeg via imageio-ffmpeg
        clip.audio.write_audiofile(
            wav_path,
            fps=target_sr,
            codec="pcm_s16le",
            verbose=False,
            logger=None
        )
        clip.close()
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    return y.astype(np.float32), sr, dur_s


def load_video_frames(path: str, fps: int = 25, as_rgb: bool = True, max_seconds: float | None = None) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Returns:
      frames: list of HxWx3 uint8 arrays
      times_s: list of timestamps (seconds) aligned to fps
      video_dur_s: duration in seconds (float)
    """
    clip = VideoFileClip(path)
    video_dur_s = float(clip.duration)

    if max_seconds is not None:
        end = min(max_seconds, video_dur_s)
        subclip = clip.subclip(0, end)
        clip_to_iter = subclip
        video_dur_s = float(end)
    else:
        clip_to_iter = clip

    frames = []
    for idx, frame in enumerate(clip_to_iter.iter_frames(fps=fps, dtype="uint8")):
        # moviepy frames are already RGB
        frames.append(frame)

    times_s = [i / float(fps) for i in range(len(frames))]
    clip.close()
    return frames, times_s, video_dur_s


def basic_info(path: str, fps: int = 25) -> dict:
    """Quick summary used by CLI and smoke tests."""
    y, sr, aud_dur = load_audio_mono(path)
    frames, times_s, vid_dur = load_video_frames(path, fps=fps)
    return {
        "audio_sr": sr,
        "audio_len_samples": int(len(y)),
        "audio_duration_s": round(aud_dur, 3),
        "frame_count": len(frames),
        "frame_fps": fps,
        "video_duration_s": round(vid_dur, 3),
        "first_ts_s": round(times_s[0], 3) if times_s else None,
        "last_ts_s": round(times_s[-1], 3) if times_s else None,
    }


def _cli():
    p = argparse.ArgumentParser(description="VeriFace media I/O sanity check")
    p.add_argument("--in", dest="inp", required=True, help="Path to video/audio file (e.g., sample.mp4)")
    p.add_argument("--fps", type=int, default=25, help="Target frame rate")
    args = p.parse_args()

    info = basic_info(args.inp, fps=args.fps)
    print("✅ Media summary")
    for k, v in info.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    _cli()
