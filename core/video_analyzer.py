import cv2
import numpy as np
from moviepy import VideoFileClip
import librosa
from pydub import AudioSegment
import scipy.signal

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
    hop_length = 512  # Poprawiono literówkę
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

    # Break into chunks (e.g., 200 ms)
    chunk_size = int(window_ms * audio.frame_rate / 1000)
    loudness = [
        np.mean(np.abs(samples[i:i+chunk_size]))
        for i in range(0, len(samples), chunk_size)
    ]
    return np.array(loudness)



def extract_audio(video_path, output_path="temp_audio.wav"):
    with VideoFileClip(video_path) as clip:
        if clip.audio is not None:
            # Wymuszamy zapis jako wav, co jest najszybsze
            clip.audio.write_audiofile(output_path, logger=None, codec='pcm_s16le')

def detect_highlights_gaming(loudness, total_duration, threshold=0.8, min_gap=15, window_ms=200):
    peaks, _ = scipy.signal.find_peaks(loudness, height=threshold)
    highlights = []
    last_end_time = -min_gap

    for p in peaks:
        t_peak = p * window_ms / 1000
        
        # W gamingu: 8s przed pikiem (akcja) i 3s po piku (reakcja/wynik)
        start = max(0, t_peak - 8)
        end = min(total_duration, t_peak + 3)
        
        # Unikaj nakładania się klipów i sprawdzaj sensowną długość
        if start >= last_end_time and (end - start) > 1:
            highlights.append((start, end))
            last_end_time = end
            
    return highlights

def make_highlight(video_path, start, end, output_path):
    # Używamy context managera 'with', aby poprawnie zamykać pliki
    with VideoFileClip(video_path) as clip:
        # Dodatkowe zabezpieczenie przed końcem filmu
        end = min(end, clip.duration)
        new_clip = clip.subclipped(start, end)
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def normalize_and_combine(energy, loudness):
		energy = energy / np.max(energy)
		loudness = loudness / np.max(loudness)

		# Align lengths (interpolate loudness to match energy)
		min_len = min(len(energy), len(loudness))
		energy = energy[:min_len]
		loudness = np.interp(np.linspace(0, len(loudness), min_len),
							np.arange(len(loudness)), loudness)
		
		combined_score = 0.2 * energy + 0.8 * loudness
		return combined_score