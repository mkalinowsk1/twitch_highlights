import cv2
import numpy as np
from moviepy import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy import TextClip, CompositeVideoClip
import librosa
from pydub import AudioSegment
import scipy.signal
from ultralytics import YOLO
import whisper
from datetime import timedelta
import torch


def get_motion_intensity(video_path):
	cap = cv2.VideoCapture(video_path)
	prev = None
	motion_scores = []
	while True:
		ret, frame = cap.read()
		if not ret: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if prev is not None:
			diff = cv2.absdiff(prev, gray)
			motion_scores.append(np.sum(diff))
		prev = gray
	cap.release()
	return np.array(motion_scores)

def get_audio_energy(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = 2048
    hop_length = 512  
    energy = np.array([
        sum(np.abs(y[i:i+frame_length])**2)
        for i in range(0, len(y), hop_length)
    ])
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    return energy, sr, hop_length

def analyze_audio_loudness(audio_path, window_ms=200):
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples())
    
    chunk_size = int(window_ms * audio.frame_rate / 1000)
    loudness = [
        np.mean(np.abs(samples[i:i+chunk_size]))
        for i in range(0, len(samples), chunk_size)
    ]
    
    loudness = np.array(loudness)
    if np.max(loudness) > 0:
        loudness = loudness / np.max(loudness)
        
    return loudness


def extract_audio(video_path, output_path="temp_audio.wav"):
    with VideoFileClip(video_path) as clip:
        if clip.audio is not None:

            clip.audio.write_audiofile(output_path, logger=None, codec='pcm_s16le')

def detect_highlights_gaming(loudness, total_duration, threshold=0.8, min_gap=15, window_ms=200):
    peaks, _ = scipy.signal.find_peaks(loudness, height=threshold)
    highlights = []
    last_end_time = -min_gap

    for p in peaks:
        t_peak = p * window_ms / 1000
        
        start = max(0, t_peak - 8)
        end = min(total_duration, t_peak + 3)
        
        if start >= last_end_time and (end - start) > 1:
            highlights.append((start, end))
            last_end_time = end
            
    return highlights

# Stare highlighty bez napisów
def make_highlight(video_path, start, end, output_path):
    with VideoFileClip(video_path) as clip:
        end = min(end, clip.duration)
        new_clip = clip.subclipped(start, end)
        new_clip.write_videofile(output_path, 
            codec="h264_nvenc", 
            audio_codec="aac",
            bitrate="8000k",
            preset="fast",
            
            ffmpeg_params=[
                "-rc", "vbr", 
                "-cq", "24", 
                "-pix_fmt", "yuv420p" 
            ]
            )

# Nowe higlighty z napisami
def make_highlight_with_subs(video_path, start, end, output_path):
    temp_sub_audio = "temp_segment.wav"
    srt_file = "temp_subs.srt"
    
    with VideoFileClip(video_path).subclipped(start, end) as segment:
        segment.audio.write_audiofile(temp_sub_audio, logger=None)
        generate_subtitles(temp_sub_audio, srt_file)
        
        srt_path_ffmpeg = srt_file.replace("\\", "/").replace(":", "\\:")
        
        segment.write_videofile(
            output_path, 
            codec="h264_nvenc", 
            audio_codec="aac",
            bitrate="8000k",
            preset="fast",
            ffmpeg_params=[
                "-vf", f"subtitles='{srt_path_ffmpeg}'",
                "-rc", "vbr", 
                "-cq", "24", 
                "-pix_fmt", "yuv420p"
            ]
        )

def normalize_and_combine(energy, loudness):
		energy = energy / np.max(energy)
		loudness = loudness / np.max(loudness)

		min_len = min(len(energy), len(loudness))
		energy = energy[:min_len]
		loudness = np.interp(np.linspace(0, len(loudness), min_len),
							np.arange(len(loudness)), loudness)
		
		combined_score = 0.2 * energy + 0.8 * loudness
		return combined_score

def calculate_auto_threshold(loudness_data, multiplier=2.0):
    if len(loudness_data) == 0:
        return 0.8
        
    mean_val = np.mean(loudness_data)
    std_val = np.std(loudness_data)

    auto_thresh = mean_val + (multiplier * std_val)
    
    return float(np.clip(auto_thresh, 0.1, 0.95))


def get_visual_activity(video_path, highlights):
    model = YOLO('yolov11x.pt')
    visual_scores = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for start, end in highlights:
        mid_frame_index = int(((start + end) / 2) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
        ret, frame = cap.read()
        
        if ret:
            results = model(frame, verbose=False)
            detection_count = len(results[0].boxes)
            visual_scores.append(detection_count)
        else:
            visual_scores.append(0)
            
    cap.release()
    return visual_scores

def get_visual_activity_score(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    
    prev_gray = None
    motion_values = []
    
    for i in range(int((end - start) * fps / 5)):
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180)) 
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_values.append(np.mean(diff))
        
        prev_gray = gray
        for _ in range(4): cap.grab()

    cap.release()
    
    if not motion_values: return 0
    return np.mean(motion_values)

def generate_subtitles(audio_path, output_srt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = whisper.load_model("base", device = device)
    result = model.transcribe(audio_path, fp16 = (device == "cuda"), language = "pl")

    with open(output_srt, "w", encoding="utf-8-sig") as f:
        for i, segment in enumerate(result['segments'], start = 1):
            start = str(timedelta(seconds = int(segment['start']))) + ",000"
            end = str(timedelta(seconds = int(segment['end']))) + ",000"
            text = segment['text'].strip()

            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    return output_srt