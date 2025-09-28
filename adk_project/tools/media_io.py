# adk_project/tools/media_io.py
# Minimal media I/O for VeriFace
# - extract frames at fixed FPS (optionally every Nth frame)
# - stream frames as (frame, ts) pairs without loading everything into memory
# - load mono audio at a fixed sample rate
# - quick CLI to sanity-check a file
from __future__ import annotations

import argparse
from typing import Tuple, List, Optional, Dict

import numpy as np
import librosa
from moviepy.editor import VideoFileClip


# ------------------------------ Audio ------------------------------

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


# ------------------------------ Video (batch) ------------------------------

def load_video_frames(
    path: str,
    fps: int = 25,
    as_rgb: bool = True,
    max_seconds: Optional[float] = None,
    frame_stride: int = 1,
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Returns:
      frames: list of HxWx3 uint8 arrays (RGB as provided by MoviePy)
      times_s: list of timestamps (seconds) aligned to source frame index / fps
      video_dur_s: duration in seconds (float)

    Notes:
      - If frame_stride > 1, we keep every Nth frame (0-based), so timestamps
        step by (frame_stride / fps).
      - MoviePy already yields RGB frames; 'as_rgb' kept for API parity.
    """
    if frame_stride < 1:
        frame_stride = 1

    clip = VideoFileClip(path)
    video_dur_s = float(clip.duration)

    # Apply temporal cap
    if max_seconds is not None:
        end_s = min(max_seconds, video_dur_s)
        subclip = clip.subclip(0, end_s)
        clip_iter = subclip
        video_dur_s = float(end_s)
    else:
        clip_iter = clip

    frames: List[np.ndarray] = []
    times_s: List[float] = []

    try:
        for idx, frame in enumerate(clip_iter.iter_frames(fps=fps, dtype="uint8")):
            if idx % frame_stride != 0:
                continue
            frames.append(frame)          # Already RGB
            times_s.append(idx / float(fps))
    finally:
        try:
            clip_iter.close()
        except Exception:
            pass
        try:
            clip.close()
        except Exception:
            pass

    return frames, times_s, video_dur_s


# ------------------------------ Video (streaming) ------------------------------

def stream_video_frames(
    path: str,
    fps: int = 25,
    frame_stride: int = 1,
    max_seconds: Optional[float] = None,
):
    """
    Streaming generator that yields (frame, timestamp_s) without storing all frames.

    Returns:
      frame_gen: generator yielding (np.ndarray HxWx3 uint8 RGB, ts_seconds)
      video_dur_s: float duration of the (sub)clip in seconds
    """
    if frame_stride < 1:
        frame_stride = 1

    clip = VideoFileClip(path)
    video_dur_s = float(clip.duration)

    if max_seconds is not None:
        end_s = min(max_seconds, video_dur_s)
        clip_iter = clip.subclip(0, end_s)
        video_dur_s = float(end_s)
    else:
        clip_iter = clip

    def _gen():
        try:
            for idx, frame in enumerate(clip_iter.iter_frames(fps=fps, dtype="uint8")):
                if idx % frame_stride != 0:
                    continue
                ts = idx / float(fps)
                yield frame, ts
        finally:
            try:
                clip_iter.close()
            except Exception:
                pass
            try:
                clip.close()
            except Exception:
                pass

    return _gen(), video_dur_s


# ------------------------------ Probing ------------------------------

def probe_media_meta(path: str) -> Dict[str, object]:
    """
    Lightweight media probe via MoviePy.
    Returns keys when available:
      {
        "duration": float seconds,
        "fps": float or None,
        "size": (width, height),
        "audio_fps": int or None,
        "audio_channels": int or None
      }
    """
    meta: Dict[str, object] = {}
    try:
        clip = VideoFileClip(path)
        try:
            meta["duration"] = float(clip.duration) if clip.duration is not None else None
        except Exception:
            meta["duration"] = None
        try:
            meta["fps"] = float(getattr(clip, "fps", None)) if getattr(clip, "fps", None) is not None else None
        except Exception:
            meta["fps"] = None
        try:
            w, h = clip.size
            meta["size"] = (int(w), int(h))
        except Exception:
            meta["size"] = None
        try:
            if clip.audio is not None:
                meta["audio_fps"] = getattr(clip.audio, "fps", None)
                meta["audio_channels"] = getattr(clip.audio, "nchannels", None)
            else:
                meta["audio_fps"] = None
                meta["audio_channels"] = None
        except Exception:
            meta["audio_fps"] = None
            meta["audio_channels"] = None
    except Exception:
        meta.setdefault("duration", None)
        meta.setdefault("fps", None)
        meta.setdefault("size", None)
        meta.setdefault("audio_fps", None)
        meta.setdefault("audio_channels", None)
    finally:
        try:
            clip.close()  # type: ignore
        except Exception:
            pass
    return meta


# ------------------------------ Utilities / CLI ------------------------------

def basic_info(path: str, fps: int = 25, frame_stride: int = 1) -> dict:
    """Quick summary used by CLI and smoke tests."""
    y, sr, aud_dur = load_audio_mono(path)
    frames, times_s, vid_dur = load_video_frames(path, fps=fps, frame_stride=frame_stride)
    info = {
        "audio_sr": sr,
        "audio_len_samples": int(len(y)),
        "audio_duration_s": round(aud_dur, 3),
        "frame_count": len(frames),
        "frame_fps": fps,
        "frame_stride": frame_stride,
        "video_duration_s": round(vid_dur, 3),
        "first_ts_s": round(times_s[0], 3) if times_s else None,
        "last_ts_s": round(times_s[-1], 3) if times_s else None,
    }
    # Enrich with probe (best-effort)
    try:
        info["probe"] = probe_media_meta(path)
    except Exception:
        pass
    return info


def _cli():
    p = argparse.ArgumentParser(description="VeriFace media I/O sanity check")
    p.add_argument("--in", dest="inp", required=True, help="Path to video/audio file (e.g., sample.mp4)")
    p.add_argument("--fps", type=int, default=25, help="Target frame rate")
    p.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame (1 = keep all)")
    args = p.parse_args()

    info = basic_info(args.inp, fps=args.fps, frame_stride=args.frame_stride)
    print("✅ Media summary")
    for k, v in info.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    _cli()
