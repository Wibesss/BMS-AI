import moviepy as mp
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import whisper
import numpy as np
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Step 1: Extract audio from the video
video = mp.VideoFileClip("video.mp4")
audio_file = video.audio
audio_file.write_audiofile("audio.wav")

# Step 2: Load the audio and compute MFCCsasdada
audio_path = 'audio.wav'
audio, sr = librosa.load(audio_path, sr=None)
mfccs = librosa.feature.mfcc(y=audio, sr=sr)

# Step 3: Normalize MFCCs and apply clustering
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T)
kmeans = KMeans(n_clusters=2, random_state=42)
speaker_labels = kmeans.fit_predict(mfccs_scaled)

# Step 4: Generate timestamps for MFCC frames
timestamps = np.linspace(0, len(audio) / sr, len(speaker_labels))

# Step 5: Transcribe the audio using Whisper
model = whisper.load_model("base", device="cuda")
result = model.transcribe(audio_path)

# Get transcription segments with start and end times
transcription_segments = result['segments']

# Step 6: Match transcription segments to speaker labels
transcription_with_speakers = []
for segment in transcription_segments:
    segment_start = segment['start']
    segment_end = segment['end']
    segment_text = segment['text']

    # Find the speaker label for this segment
    start_idx = np.searchsorted(timestamps, segment_start, side="left")
    end_idx = np.searchsorted(timestamps, segment_end, side="right")

    # Determine the dominant speaker for this segment
    dominant_speaker = np.bincount(speaker_labels[start_idx:end_idx]).argmax()
    transcription_with_speakers.append((dominant_speaker, segment_text))

# Step 7: Print the transcription alternating speakers
for speaker, text in transcription_with_speakers:
    print(f"Speaker {speaker}: {text}")
