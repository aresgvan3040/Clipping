# HikeClipper — Full-Stack Video Hike & Facecam Clipper

HikeClipper analyzes any video link (YouTube, Twitch, etc.) or uploaded MP4, detects **viewer-hike / high-engagement moments**, and cuts **30-second clips** (`20s before hike` + `10s after hike`) into mobile- and desktop-ready **MP4** files.

## Features
- **Hike Detection (2–10 Clips)**: Uses YouTube's Most Replayed heatmap data when available, with automatic FFmpeg audio-energy spike fallback.
- **Exact 20s + 10s Window**: Automatically slices 20 seconds before each hike peak and 10 seconds after (30s total), with manual fine-tuning.
- **Aspect Ratio Switcher**: Export in `9:16` (Shorts/Reels/TikTok), `16:9` (YouTube), `1:1` (Square), or `4:5` (Portrait).
- **Facecam Top-Split & Custom Crop Permission**: Visually select the exact Facecam region and Main content region on the source video; places Facecam on top and Main video below.
- **Auto-Caption Generator**: Extracts video transcripts/subtitles (or lets you edit/add lines) and burns styled captions into the final MP4.
- **Universal MP4 Download**: Encoded with `H.264 (yuv420p) + AAC + faststart` for seamless playback on iOS, Android, macOS, and Windows.

## Quick Start
### Prerequisites
- Python 3.9+
- `ffmpeg` and `ffprobe` installed and on your `PATH`

### Run on macOS / Linux
```bash
chmod +x start.sh
./start.sh
```

### Run on Windows
Double-click `start.bat`, or run:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Then open **http://127.0.0.1:5050** in your browser (or from your phone on the same Wi-Fi using your computer's local IP, e.g. `http://192.168.1.x:5050`).
