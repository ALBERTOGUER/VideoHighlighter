import os
import subprocess

from modules.media.lut import lut_filter
from modules.system.app_paths import ffmpeg_exe
from modules.system.encoder_select import encoder_chain


def cut_video(video_path, start_time, end_time, output_path, mode="gpu", lut_path=None):
    """Cut [start_time, end_time] out of video_path into output_path.

    `mode`:
      - "cpu": re-encode with libx265 (HEVC/VR) or libx264 — VR-safe, slow.
      - "gpu" (default): re-encode with the fastest hardware encoder, CPU
        fallback. Fast, but hardware HEVC may not play in some VR players.

    `lut_path`: optional 3D LUT (.cube…) applied before the pixel format
    conversion, so a log-profile source is graded at its full bit depth.

    Pixel format is normalized to yuv420p so 10-bit VR sources don't break the
    encoders."""
    duration = end_time - start_time
    # normalize (VR is often 10-bit). setpts: after a fast seek the first frame
    # can land one frame past zero and be dropped, which the stream-copy concat
    # of the reel then shows as a one-frame gap at every join.
    vf = "format=yuv420p,setpts=PTS-STARTPTS"
    if lut_path:
        vf = f"{lut_filter(lut_path)},{vf}"
    last_err = "unknown error"
    for enc, vargs in encoder_chain(video_path, mode=mode):
        cmd = [
            ffmpeg_exe(), "-y", "-v", "error",
            "-ss", str(start_time),   # fast seek before decoding
            "-i", video_path,
            "-t", str(duration),
            "-fflags", "+genpts",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-vf", vf,
        ] + vargs + [
            "-c:a", "aac",            # re-encode audio
            "-b:a", "128k",
            "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Clip saved: {output_path} [{enc}]")
            return
        last_err = (result.stderr or "").strip()[-500:] or "unknown error"
        # Include ffmpeg's own message: the return code alone (e.g. QSV's
        # 0xB1B1B1AB) tells you nothing, and when the whole chain fails this is
        # the only clue the user ever sees.
        first_line = last_err.splitlines()[0] if last_err.splitlines() else last_err
        print(f"⚠️ cut_video {enc} failed (rc={result.returncode}): {first_line}; "
              + ("trying next encoder" if enc != "libx264" else "no fallback left"))
    raise RuntimeError(f"cut_video failed for {output_path}: {last_err}")
