import moviepy as mp
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import whisper
import numpy as np
import scipy.signal
import os
from collections import Counter
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def createTranscript(numOfSpeaker, videoPath):
    numOfSpeakers = int(numOfSpeaker)
    video = mp.VideoFileClip(videoPath)
    audio_file = video.audio
    
    audioName = os.path.splitext(videoPath)[0]

    audio_file.write_audiofile(f"{audioName}.wav")

    audio_path = f"{audioName}.wav"
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)

    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    kmeans = KMeans(n_clusters=numOfSpeakers, random_state=42)
    speaker_labels = kmeans.fit_predict(mfccs_scaled)

    smoothed_speakers = scipy.signal.medfilt(speaker_labels, kernel_size=5) 

    timestamps = np.linspace(0, len(audio) / sr, len(speaker_labels))

    model = whisper.load_model("base", device="cuda")
    result = model.transcribe(audio_path)

    transcription_segments = result['segments']
    
    transcription_with_speakers = []
    for segment in transcription_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        segment_text = segment['text']

        start_idx = np.searchsorted(timestamps, segment_start, side="left")
        end_idx = np.searchsorted(timestamps, segment_end, side="right")

        relevant_speakers = smoothed_speakers[start_idx:end_idx]

        if len(relevant_speakers) > 0:
            speaker_counts = Counter(relevant_speakers)
            dominant_speaker = speaker_counts.most_common(1)[0][0] + 1
        else:
            dominant_speaker = 1

        transcription_with_speakers.append((dominant_speaker, segment_text))

    transcript = ""
    currentSpeaker = 1
    for speaker, text in transcription_with_speakers:
        if currentSpeaker == speaker:
            transcript += f" {text}"
        elif (text.lstrip())[0].islower():
            transcript += f" {text}"
        else:
            transcript += f"\n #Person{speaker}#: {text}"
        currentSpeaker = speaker
        
    return transcript