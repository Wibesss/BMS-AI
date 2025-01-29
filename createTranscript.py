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
    # Step 1: Extract audio from the video
    video = mp.VideoFileClip(videoPath)
    audio_file = video.audio
    
    audioName = os.path.splitext(videoPath)[0]
    print(audioName)  # Output: video

    audio_file.write_audiofile(f"{audioName}.wav")

    # Step 2: Load the audio and compute MFCCsasdada
    audio_path = f"{audioName}.wav"
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)

    # Step 3: Normalize MFCCs and apply clustering
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    kmeans = KMeans(n_clusters=numOfSpeaker, random_state=42)
    speaker_labels = kmeans.fit_predict(mfccs_scaled)

    # Apply median filter for smoothing speaker labels
    smoothed_speakers = scipy.signal.medfilt(speaker_labels, kernel_size=5)  # Adjust kernel size if needed

    # Step 4: Generate timestamps for MFCC frames
    timestamps = np.linspace(0, len(audio) / sr, len(speaker_labels))

    # Step 5: Transcribe the audio using Whisper
    model = whisper.load_model("base", device="cuda")
    result = model.transcribe(audio_path)

    # Get transcription segments with start and end times
    transcription_segments = result['segments']

    # Step 6: Match transcription segments to speaker labels
    from collections import Counter

# Step 6: Match transcription segments to speaker labels
    # Step 6: Match transcription segments to speaker labels
    transcription_with_speakers = []
    for segment in transcription_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        segment_text = segment['text']

        # Find the closest MFCC indices for the start and end times
        start_idx = np.searchsorted(timestamps, segment_start, side="left")
        end_idx = np.searchsorted(timestamps, segment_end, side="right")

        # Get speaker labels in this time window (Use smoothed version)
        relevant_speakers = smoothed_speakers[start_idx:end_idx]

        if len(relevant_speakers) > 0:
            # Use majority voting to determine the dominant speaker
            speaker_counts = Counter(relevant_speakers)
            dominant_speaker = speaker_counts.most_common(1)[0][0] + 1  # Start from 1
        else:
            dominant_speaker = 1  # Default to Speaker 1 if no data is available

        transcription_with_speakers.append((dominant_speaker, segment_text))


    # Step 7: Print the transcription alternating speakers

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