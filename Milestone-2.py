# Generated from: Milestone-2.ipynb
# Converted at: 2026-03-15T16:38:21.654Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
     (os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

ROOT = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'
STEMS_PATH = os.path.join(ROOT, 'genres_stems')

GENRES = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

jazz_path = os.path.join(STEMS_PATH, "jazz")

durations = []

for song in tqdm(os.listdir(jazz_path)):
    song_path = os.path.join(jazz_path, song)
    
    if os.path.isdir(song_path):
        for stem in os.listdir(song_path):
            file_path = os.path.join(song_path, stem)
            
            if os.path.exists(file_path):
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    durations.append(len(y) / sr)
                except:
                    continue

print("Answer Q1 (Mean Duration Jazz):", np.mean(durations))

sample_rates = set()

for g in GENRES:
    genre_path = os.path.join(STEMS_PATH, g)
    
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        
        for stem in os.listdir(song_path):
            file_path = os.path.join(song_path, stem)
            
            if os.path.exists(file_path):
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    sample_rates.add(sr)
                except:
                    continue

print("Answer Q2 (Unique Sample Rates):", sorted(list(sample_rates)))

corrupted_count = 0

for g in GENRES:
    genre_path = os.path.join(STEMS_PATH, g)
    
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        
        for stem in os.listdir(song_path):
            file_path = os.path.join(song_path, stem)
            
            if os.path.exists(file_path):
                if os.path.getsize(file_path) == 0:
                    corrupted_count += 1

print("Answer Q3 (Corrupted Files):", corrupted_count)

peak_db_values = []

for g in GENRES:
    genre_path = os.path.join(STEMS_PATH, g)
    
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        
        vocal_path = os.path.join(song_path, "vocals.wav")
        
        if os.path.exists(vocal_path):
            try:
                y, sr = librosa.load(vocal_path, sr=None)
                
                if len(y) > 0:
                    peak = np.max(np.abs(y))
                    peak_db = 20 * np.log10(peak + 1e-10)
                    peak_db_values.append(peak_db)
                    
            except:
                continue

print("Answer Q4 (Avg Peak dB Vocals):", np.mean(peak_db_values))

blues_path = os.path.join(STEMS_PATH, "blues")

centroids = []

for song in os.listdir(blues_path):
    song_path = os.path.join(blues_path, song)
    
    for fname in ['other.wav', 'others.wav']:
        file_path = os.path.join(song_path, fname)
        
        if os.path.exists(file_path):
            try:
                y, sr = librosa.load(file_path, sr=22050)
                
                if len(y) > 0:
                    spec_cent = np.mean(
                        librosa.feature.spectral_centroid(y=y, sr=sr)
                    )
                    centroids.append(spec_cent)
            except:
                continue

print("Answer Q5 (Mean Spectral Centroid Blues):", np.mean(centroids))

genre_centroids = {}

for g in GENRES:
    centroids = []
    genre_path = os.path.join(STEMS_PATH, g)
    
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        
        for fname in ['other.wav', 'others.wav']:
            file_path = os.path.join(song_path, fname)
            
            if os.path.exists(file_path):
                try:
                    y, sr = librosa.load(file_path, sr=22050)
                    
                    if len(y) > 0:
                        spec_cent = np.mean(
                            librosa.feature.spectral_centroid(y=y, sr=sr)
                        )
                        centroids.append(spec_cent)
                except:
                    continue
    
    if len(centroids) > 0:
        genre_centroids[g] = np.mean(centroids)

print("All Genre Means:", genre_centroids)
print("Answer Q6 (Highest Centroid Genre):",
      max(genre_centroids, key=genre_centroids.get))

silence_count = 0

for g in GENRES:
    genre_path = os.path.join(STEMS_PATH, g)
    
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        
        for stem in os.listdir(song_path):
            file_path = os.path.join(song_path, stem)
            
            if os.path.exists(file_path):
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    first_half_sec = y[:int(0.5 * sr)]
                    
                    if np.max(np.abs(first_half_sec)) < 1e-4:
                        silence_count += 1
                        
                except:
                    continue

print("Answer Q7 (Silence Count):", silence_count)



def extract_features_safe(song_path):
    
    # auto detect stem name
    possible_files = ['other.wav', 'others.wav']
    file_path = None
    
    for fname in possible_files:
        temp_path = os.path.join(song_path, fname)
        if os.path.exists(temp_path):
            file_path = temp_path
            break
    
    if file_path is None:
        return None
    
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=10)
        
        if len(y) == 0:
            return None
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        return [float(tempo), spec_cent, zcr, rolloff]
    
    except:
        return None

data = []

for g in GENRES:
    gp = os.path.join(STEMS_PATH, g)
    songs = [s for s in os.listdir(gp) if os.path.isdir(os.path.join(gp, s))]
    
    for s in songs[:50]:   # speed ke liye
        data.append({'path': os.path.join(gp, s), 'genre': g})

df = pd.DataFrame(data)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['genre'],
    random_state=42
)

X_train = []
y_train = []

for path, genre in zip(train_df['path'], train_df['genre']):
    features = extract_features_safe(path)
    if features is not None:
        X_train.append(features)
        y_train.append(genre)

X_val = []
y_val = []

for path, genre in zip(val_df['path'], val_df['genre']):
    features = extract_features_safe(path)
    if features is not None:
        X_val.append(features)
        y_val.append(genre)

X_train = np.array(X_train)
X_val = np.array(X_val)

print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

macro_f1 = f1_score(y_val, y_pred, average='macro')
print("Answer Q8 (Validation Macro F1):", macro_f1)

cr = classification_report(y_val, y_pred, output_dict=True)

print("Answer Q9 (Precision hiphop):", cr['hiphop']['precision'])

print("Answer Q10 (Recall pop):", cr['pop']['recall'])

accuracy = np.mean(y_pred == y_val)
print("Answer Q11 (Accuracy):", accuracy)

cm = confusion_matrix(y_val, y_pred, labels=GENRES)

tp_dict = {}

for i, genre in enumerate(GENRES):
    TP = cm[i, i]
    tp_dict[genre] = TP

print("Answer Q12 (Highest TP Genre):", max(tp_dict, key=tp_dict.get))

fn_dict = {}

for i, genre in enumerate(GENRES):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    fn_dict[genre] = FN

print("Answer Q13 (Lowest FN Genre):", min(fn_dict, key=fn_dict.get))