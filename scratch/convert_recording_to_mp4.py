"""
convert_recording_to_mp4.py
===========================
Extracts frames from demo_recording.webp and encodes a crisp H.264 MP4 video
at a comfortable presentation frame rate (~3-4 fps for animation, or 60-90s duration).
"""

import os
from PIL import Image, ImageSequence
import imageio
import imageio_ffmpeg

input_path = "demo_recording.webp"
output_path = "demo_recording.mp4"

print(f"Reading {input_path}...")
im = Image.open(input_path)
n_frames = im.n_frames
print(f"Total frames: {n_frames}, Resolution: {im.size}")

# Target presentation duration ~60 to 75 seconds -> fps ~ 4
target_fps = 4

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg at: {ffmpeg_exe}")

writer = imageio.get_writer(
    output_path,
    fps=target_fps,
    codec='libx264',
    format='FFMPEG',
    ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '20']
)

frame_idx = 0
for frame in ImageSequence.Iterator(im):
    # Ensure RGB (width and height divisible by 2 for H.264)
    rgb_frame = frame.convert('RGB')
    w, h = rgb_frame.size
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    if (new_w, new_h) != (w, h):
        rgb_frame = rgb_frame.crop((0, 0, new_w, new_h))
    
    import numpy as np
    writer.append_data(np.array(rgb_frame))
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"  Processed {frame_idx}/{n_frames} frames...")

writer.close()
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
duration_sec = n_frames / target_fps
print(f"SUCCESS: Exported {output_path} ({duration_sec:.1f}s, {file_size_mb:.2f} MB)")
