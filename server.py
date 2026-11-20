import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="HikeClipper Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}

RATIO_DIMS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}


class BoxRegion(BaseModel):
    x: float = Field(0.0, ge=0.0, le=1.0)
    y: float = Field(0.0, ge=0.0, le=1.0)
    w: float = Field(1.0, gt=0.0, le=1.0)
    h: float = Field(1.0, gt=0.0, le=1.0)


class CaptionLine(BaseModel):
    start: float
    end: float
    text: str


class AnalyzeRequest(BaseModel):
    url: str
    max_clips: int = Field(6, ge=2, le=10)


class RenderRequest(BaseModel):
    job_id: str
    clip_id: str
    start: float
    end: float
    ratio: str = "9:16"
    facecam_enabled: bool = False
    facecam_box: BoxRegion = Field(default_factory=lambda: BoxRegion(x=0.72, y=0.68, w=0.26, h=0.28))
    main_box: BoxRegion = Field(default_factory=lambda: BoxRegion(x=0.0, y=0.0, w=1.0, h=1.0))
    captions_enabled: bool = False
    captions: List[CaptionLine] = Field(default_factory=list)
    caption_style: str = "bold"


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def get_video_info(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))
    width, height = 1920, 1080
    has_audio = False
    for st in data.get("streams", []):
        if st.get("codec_type") == "video":
            width = int(st.get("width", 1920))
            height = int(st.get("height", 1080))
        elif st.get("codec_type") == "audio":
            has_audio = True
    return {"duration": duration, "width": width, "height": height, "has_audio": has_audio}


def detect_audio_hikes(video_path: Path, duration: float) -> Dict[int, float]:
    """Analyze full-video audio RMS energy per second."""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-af",
        "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    buckets: Dict[int, List[float]] = {}
    cur_t = 0.0
    for line in proc.stdout.splitlines():
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([0-9.]+)", line)
            if m:
                cur_t = float(m.group(1))
        elif "lavfi.astats.Overall.RMS_level=" in line:
            val_str = line.split("=", 1)[1].strip()
            try:
                db = float(val_str)
                if math.isinf(db) or db < -80:
                    db = -80.0
            except ValueError:
                db = -80.0
            score = max(0.0, min(1.0, (db + 60.0) / 50.0))
            buckets.setdefault(int(cur_t), []).append(score)
    return {sec: sum(vals) / len(vals) for sec, vals in buckets.items()}


def detect_motion_hikes(video_path: Path, duration: float) -> Dict[int, float]:
    """Analyze full-video visual movement & scene-change intensity per second."""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "fps=2,scale=320:-1,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    buckets: Dict[int, List[float]] = {}
    cur_t = 0.0
    for line in proc.stdout.splitlines():
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([0-9.]+)", line)
            if m:
                cur_t = float(m.group(1))
        elif "lavfi.scene_score=" in line:
            try:
                val = float(line.split("=", 1)[1].strip())
            except ValueError:
                val = 0.0
            # Amplify typical 0.02-0.40 motion/scene scores into 0..1
            score = max(0.0, min(1.0, val * 3.0))
            buckets.setdefault(int(cur_t), []).append(score)
    return {sec: max(vals) for sec, vals in buckets.items()}


def analyze_full_video_hikes(
    video_path: Path,
    duration: float,
    yt_heatmap: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, float]]:
    """Combine YouTube heatmap (if any) + full-video audio spikes + visual motion."""
    total_secs = max(1, int(math.ceil(duration)))
    audio_map = detect_audio_hikes(video_path, duration)
    motion_map = detect_motion_hikes(video_path, duration)

    yt_map: Dict[int, float] = {}
    if yt_heatmap:
        for pt in yt_heatmap:
            yt_map[int(pt["t"])] = pt["score"]

    raw_scores: List[float] = []
    for sec in range(total_secs):
        a = audio_map.get(sec, 0.2)
        m = motion_map.get(sec, 0.15)
        if yt_map:
            # Nearest heatmap bucket
            y = yt_map.get(sec, yt_map.get(max(0, sec - 1), 0.3))
            combined = 0.60 * y + 0.25 * a + 0.15 * m
        else:
            combined = 0.55 * a + 0.45 * m
        raw_scores.append(combined)

    # 3-second smoothing window so sustained action peaks rank highest
    smoothed: List[Dict[str, float]] = []
    for sec in range(total_secs):
        lo = max(0, sec - 1)
        hi = min(total_secs, sec + 2)
        avg = sum(raw_scores[lo:hi]) / (hi - lo)
        smoothed.append({"t": float(sec), "score": round(min(1.0, avg), 3)})
    return smoothed


def pick_hike_clips(
    heatmap_points: List[Dict[str, float]],
    duration: float,
    max_clips: int,
    source_type: str,
) -> List[Dict[str, Any]]:
    """Pick 2 to 10 non-overlapping hike peaks and build 30s clips (20s before + 10s after)."""
    target_count = max(2, min(10, max_clips))
    if duration <= 30.0:
        return [
            {
                "id": "clip_1",
                "title": "Hike Peak #1",
                "peak_time": round(duration * 0.66, 2),
                "start": 0.0,
                "end": round(duration, 2),
                "score": 1.0,
                "source": source_type,
            }
        ]

    sorted_pts = sorted(heatmap_points, key=lambda x: x["score"], reverse=True)
    selected_peaks: List[Dict[str, float]] = []
    min_spacing = 28.0 if duration > target_count * 32 else max(12.0, duration / (target_count + 1))

    for pt in sorted_pts:
        t = pt["t"]
        if all(abs(t - sp["t"]) >= min_spacing for sp in selected_peaks):
            selected_peaks.append(pt)
        if len(selected_peaks) >= target_count:
            break

    # Ensure at least 2 clips if video >= 35s
    if len(selected_peaks) < 2 and duration >= 35.0:
        fallback_times = [duration * 0.35, duration * 0.72]
        for ft in fallback_times:
            if all(abs(ft - sp["t"]) >= 15.0 for sp in selected_peaks):
                selected_peaks.append({"t": round(ft, 2), "score": 0.75})
            if len(selected_peaks) >= 2:
                break

    selected_peaks.sort(key=lambda x: x["t"])
    clips: List[Dict[str, Any]] = []
    for idx, sp in enumerate(selected_peaks, start=1):
        peak = sp["t"]
        start = max(0.0, peak - 20.0)
        end = start + 30.0
        if end > duration:
            end = duration
            start = max(0.0, end - 30.0)
        clips.append(
            {
                "id": f"clip_{idx}",
                "title": f"Hike #{idx} ({int(peak // 60)}:{int(peak % 60):02d})",
                "peak_time": round(peak, 2),
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(sp["score"], 3),
                "source": source_type,
            }
        )
    return clips


def parse_vtt_subtitles(vtt_path: Path) -> List[Dict[str, Any]]:
    if not vtt_path.exists():
        return []
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    entries: List[Dict[str, Any]] = []
    time_re = re.compile(r"(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})")

    def to_sec(h: Optional[str], m: str, s: str, ms: str) -> float:
        hours = int(h[:-1]) if h else 0
        return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = time_re.search(ln)
            if m:
                start = to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
                end = to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
                caption_txt = " ".join(lines[i + 1 :])
                caption_txt = re.sub(r"<[^>]+>", "", caption_txt).strip()
                if caption_txt and end > start:
                    entries.append({"start": round(start, 2), "end": round(end, 2), "text": caption_txt})
                break
    return entries


def slice_captions_for_clip(all_subs: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    sliced: List[Dict[str, Any]] = []
    for sub in all_subs:
        if sub["end"] > start and sub["start"] < end:
            s = max(0.0, sub["start"] - start)
            e = min(end - start, sub["end"] - start)
            if e - s > 0.15:
                sliced.append({"start": round(s, 2), "end": round(e, 2), "text": sub["text"]})
    return sliced


def write_ass_file(captions: List[CaptionLine], style_name: str, width: int, height: int, out_path: Path) -> None:
    font_size = max(36, int(width * 0.052))
    margin_v = int(height * 0.12)
    styles = {
        "bold": ("Arial Black", font_size, "&H00FFFFFF", "&H00000000", 3, 2, 1),
        "classic": ("Arial", int(font_size * 0.85), "&H00FFFFFF", "&H80000000", 1, 0, 3),
        "neon": ("Arial Black", font_size, "&H0000F5D4", "&H00120826", 3, 2, 1),
        "minimal": ("Helvetica", int(font_size * 0.82), "&H00F8FAFC", "&H990F172A", 0, 0, 3),
    }
    font, fsize, prim_col, back_col, outline, shadow, border_style = styles.get(style_name, styles["bold"])

    def fmt_ass_time(sec: float) -> str:
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int((sec - int(sec)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fsize},{prim_col},&H000000FF,&H00000000,{back_col},-1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for cap in captions:
        txt = cap.text.replace("\n", "\\N").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{fmt_ass_time(cap.start)},{fmt_ass_time(cap.end)},Default,,0,0,0,,{txt}")
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def range_file_response(file_path: Path, request: Request, download_name: Optional[str] = None) -> StreamingResponse:
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
    }
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'

    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            chunk_size = end - start + 1

            def iter_range():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        data = f.read(min(1024 * 256, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(chunk_size)
            return StreamingResponse(iter_range(), status_code=206, headers=headers)

    def iter_full():
        with open(file_path, "rb") as f:
            while True:
                data = f.read(1024 * 256)
                if not data:
                    break
                yield data

    headers["Content-Length"] = str(file_size)
    return StreamingResponse(iter_full(), status_code=200, headers=headers)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    html = index_path.read_text(encoding="utf-8")
    inject = "<script>window.__HIKE_SERVER__ = true;</script></head>"
    if "</head>" in html:
        html = html.replace("</head>", inject, 1)
    return HTMLResponse(html)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ffmpeg": has_tool("ffmpeg"),
        "ffprobe": has_tool("ffprobe"),
        "yt_dlp": has_tool("yt-dlp"),
    }


@app.post("/api/analyze")
async def analyze_link(req: AnalyzeRequest):
    if not has_tool("yt-dlp"):
        raise HTTPException(status_code=500, detail="yt-dlp is not installed on the server.")
    if not has_tool("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg is not installed on the server.")

    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_template = str(job_dir / "source.%(ext)s")

    common_yt_args = [
        "--no-playlist",
        "--no-warnings",
        "--geo-bypass",
        "--js-runtimes",
        "node",
        "--extractor-args",
        "youtube:player_client=android_vr,android,ios,tv",
    ]
    cookies_env = os.environ.get("YOUTUBE_COOKIES", "").strip()
    cookies_file = job_dir / "cookies.txt"
    if cookies_env:
        cookies_file.write_text(cookies_env, encoding="utf-8")
        common_yt_args.extend(["--cookies", str(cookies_file)])
    elif (BASE_DIR / "cookies.txt").exists():
        common_yt_args.extend(["--cookies", str(BASE_DIR / "cookies.txt")])

    # 1) Fetch metadata & heatmap JSON
    meta_cmd = ["yt-dlp", *common_yt_args, "--dump-single-json", req.url]
    meta_proc = subprocess.run(meta_cmd, capture_output=True, text=True)
    if meta_proc.returncode != 0:
        err_msg = (meta_proc.stderr or meta_proc.stdout or "Unknown yt-dlp error").strip()
        raise HTTPException(status_code=400, detail=f"YouTube analyze failed: {err_msg[:350]}")
    meta = json.loads(meta_proc.stdout)

    # 2) Download MP4 (flexible format selector + merge to mp4)
    dl_cmd = [
        "yt-dlp",
        *common_yt_args,
        "-f",
        "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        video_template,
        req.url,
    ]
    dl_proc = subprocess.run(dl_cmd, capture_output=True, text=True)
    if dl_proc.returncode != 0:
        err_msg = (dl_proc.stderr or dl_proc.stdout or "Unknown download error").strip()
        raise HTTPException(status_code=400, detail=f"Video download failed: {err_msg[:350]}")

    # Best-effort subtitle download
    sub_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "en.*",
        "--sub-format",
        "vtt",
        "-o",
        video_template,
        req.url,
    ]
    subprocess.run(sub_cmd, capture_output=True, text=True)

    mp4_files = list(job_dir.glob("source*.mp4"))
    if not mp4_files:
        any_vids = [p for p in job_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if not any_vids:
            raise HTTPException(status_code=500, detail="Downloaded video file not found.")
        source_path = any_vids[0]
    else:
        source_path = mp4_files[0]

    info = get_video_info(source_path)
    duration = info["duration"] or float(meta.get("duration") or 0)

    heatmap_raw = meta.get("heatmap") or []
    yt_pts = [
        {"t": round((pt["start_time"] + pt["end_time"]) / 2.0, 2), "score": round(float(pt["value"]), 3)}
        for pt in heatmap_raw
    ] if heatmap_raw else None
    heatmap_pts = analyze_full_video_hikes(source_path, duration, yt_pts)
    source_type = "youtube_heatmap+motion" if yt_pts else "audio+motion"

    clips = pick_hike_clips(heatmap_pts, duration, req.max_clips, source_type)

    vtt_files = list(job_dir.glob("*.vtt"))
    all_subs = parse_vtt_subtitles(vtt_files[0]) if vtt_files else []
    for clip in clips:
        clip["captions"] = slice_captions_for_clip(all_subs, clip["start"], clip["end"])

    JOBS[job_id] = {
        "job_id": job_id,
        "title": meta.get("title", "Analyzed Video"),
        "source_path": str(source_path),
        "info": info,
        "heatmap": heatmap_pts,
        "clips": clips,
    }
    return {
        "job_id": job_id,
        "title": JOBS[job_id]["title"],
        "duration": duration,
        "width": info["width"],
        "height": info["height"],
        "heatmap_source": source_type,
        "heatmap": heatmap_pts,
        "clips": clips,
        "video_url": f"/api/video/{job_id}",
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), max_clips: int = Form(6)):
    if not has_tool("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg is not installed on the server.")
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "upload.mp4").suffix or ".mp4"
    source_path = job_dir / f"source{ext}"
    with open(source_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_video_info(source_path)
    duration = info["duration"]
    heatmap_pts = analyze_full_video_hikes(source_path, duration, None)
    clips = pick_hike_clips(heatmap_pts, duration, max_clips, "audio+motion")
    for clip in clips:
        clip["captions"] = []

    JOBS[job_id] = {
        "job_id": job_id,
        "title": file.filename or "Uploaded Video",
        "source_path": str(source_path),
        "info": info,
        "heatmap": heatmap_pts,
        "clips": clips,
    }
    return {
        "job_id": job_id,
        "title": JOBS[job_id]["title"],
        "duration": duration,
        "width": info["width"],
        "height": info["height"],
        "heatmap_source": "audio+motion",
        "heatmap": heatmap_pts,
        "clips": clips,
        "video_url": f"/api/video/{job_id}",
    }


@app.get("/api/video/{job_id}")
async def stream_source_video(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return range_file_response(Path(job["source_path"]), request)


@app.post("/api/render")
async def render_clip(req: RenderRequest):
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    source_path = Path(job["source_path"])
    job_dir = JOBS_DIR / req.job_id
    out_name = f"{req.clip_id}_{req.ratio.replace(':', 'x')}.mp4"
    out_path = job_dir / out_name

    target_w, target_h = RATIO_DIMS.get(req.ratio, (1080, 1920))
    duration = max(1.0, req.end - req.start)

    filter_parts: List[str] = []
    if req.facecam_enabled:
        top_h = int(round(target_h * 0.35 / 2.0)) * 2
        bot_h = target_h - top_h
        fc = req.facecam_box
        mn = req.main_box
        filter_parts.append(
            f"[0:v]crop=iw*{fc.w:.4f}:ih*{fc.h:.4f}:iw*{fc.x:.4f}:ih*{fc.y:.4f},"
            f"scale={target_w}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{top_h},setsar=1[top]"
        )
        filter_parts.append(
            f"[0:v]crop=iw*{mn.w:.4f}:ih*{mn.h:.4f}:iw*{mn.x:.4f}:ih*{mn.y:.4f},"
            f"scale={target_w}:{bot_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{bot_h},setsar=1[bot]"
        )
        filter_parts.append("[top][bot]vstack=inputs=2[stacked]")
        v_label = "[stacked]"
    else:
        mn = req.main_box
        filter_parts.append(
            f"[0:v]crop=iw*{mn.w:.4f}:ih*{mn.h:.4f}:iw*{mn.x:.4f}:ih*{mn.y:.4f},"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},setsar=1[framed]"
        )
        v_label = "[framed]"

    if req.captions_enabled and req.captions:
        ass_path = job_dir / f"{req.clip_id}.ass"
        write_ass_file(req.captions, req.caption_style, target_w, target_h, ass_path)
        filter_parts.append(f"{v_label}ass={ass_path.name}[outv]")
        v_label = "[outv]"

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{req.start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        v_label,
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-profile:v",
        "main",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(job_dir))
    if proc.returncode != 0 or not out_path.exists():
        raise HTTPException(status_code=500, detail=f"FFmpeg render failed: {proc.stderr[-500:]}")

    return {
        "clip_id": req.clip_id,
        "filename": out_name,
        "play_url": f"/api/clip/{req.job_id}/{out_name}",
        "download_url": f"/api/clip/{req.job_id}/{out_name}?download=1",
    }


@app.get("/api/clip/{job_id}/{filename}")
async def serve_clip(job_id: str, filename: str, request: Request, download: int = 0):
    clip_path = (JOBS_DIR / job_id / filename).resolve()
    if not clip_path.exists() or JOBS_DIR.resolve() not in clip_path.parents:
        raise HTTPException(status_code=404, detail="Clip not found")
    return range_file_response(clip_path, request, download_name=filename if download else None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5050)
