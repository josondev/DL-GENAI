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

# =========================
# CELL 1 — INIT WANDB
# =========================

import wandb

run = wandb.init(
    entity="23f2002424-shiv-nadar-university",
    project="23f2002424-t12026",
    name="milestone1-dataset-processing",
    config={
        "dataset": "messy_mashup",
        "sample_rate": 22050,
        "duration_threshold": 5.0,
        "top_db": 20,
        "validation_split": 0.17
    }
)

# =========================
# CELL 2 — DATASET + LOGGING
# =========================

import os
import glob
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


DATA_ROOT = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/genres_stems"
SR = 22050
DURATION = 5.0
TOP_DB = 20

random.seed(67)
np.random.seed(67)

GENRES = sorted([
    g for g in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, g))
])

STEM_KEYS = ['drums', 'vocals', 'bass', 'other']

STEMS = {
    'drums.wav': 'drums',
    'vocals.wav': 'vocals',
    'bass.wav': 'bass',
    'other.wav': 'other'
}


# ====================================
# DATASET BUILDING
# ====================================

def build_dataset(root_dir, val_split=0.17, seed=42):

    train_dataset = {g: {k: [] for k in STEM_KEYS} for g in GENRES}
    val_dataset   = {g: {k: [] for k in STEM_KEYS} for g in GENRES}

    rng = random.Random(seed)

    corrupted_count = 0
    less_5_0491MB = 0
    greater_5_0493MB = 0

    for genre in GENRES:

        genre_path = os.path.join(root_dir, genre)
        songs = sorted(os.listdir(genre_path))
        valid_songs = []

        for song in songs:

            song_path = os.path.join(genre_path, song)
            stem_files = []

            for stem_file in STEMS:

                fpath = os.path.join(song_path, stem_file)

                if not os.path.exists(fpath):
                    break

                size = os.path.getsize(fpath)

                if size < 4 * 1024:
                    corrupted_count += 1

                if size < 5.0491 * 1024 * 1024:
                    less_5_0491MB += 1

                if size > 5.0493 * 1024 * 1024:
                    greater_5_0493MB += 1

                stem_files.append(fpath)

            if len(stem_files) == 4:
                valid_songs.append(song_path)

        rng.shuffle(valid_songs)

        split_idx = int(len(valid_songs) * (1 - val_split))

        train_songs = valid_songs[:split_idx]
        val_songs   = valid_songs[split_idx:]

        for s in train_songs:
            for stem_file in STEMS:
                train_dataset[genre][STEMS[stem_file]].append(
                    os.path.join(s, stem_file)
                )

        for s in val_songs:
            for stem_file in STEMS:
                val_dataset[genre][STEMS[stem_file]].append(
                    os.path.join(s, stem_file)
                )

    print("\n--- Q1 ---")
    print("Corrupted + (<5.0491MB):", corrupted_count + less_5_0491MB)

    print("\n--- Q2 ---")
    print("Absolute difference:",
          abs(greater_5_0493MB - less_5_0491MB))

    print("\n--- Q3 ---")

    reggae_train_drums = len(train_dataset['reggae']['drums'])
    country_val_vocals = len(val_dataset['country']['vocals'])

    q3_val = abs(reggae_train_drums - country_val_vocals)

    print("Absolute difference:", q3_val)

    # WANDB LOGGING
    wandb.log({
        "corrupted_plus_small_files": corrupted_count + less_5_0491MB,
        "size_difference": abs(greater_5_0493MB - less_5_0491MB),
        "q3_dataset_difference": q3_val
    })

    return train_dataset, val_dataset


tr, val = build_dataset(DATA_ROOT)


# ====================================
# SILENCE DETECTION
# ====================================

def find_long_silences(dataset_dict,
                       sr=SR,
                       threshold_sec=DURATION,
                       top_db=TOP_DB):

    records = []

    for genre in dataset_dict:
        for stem in dataset_dict[genre]:

            for file_path in tqdm(dataset_dict[genre][stem], leave=False):

                y, _ = librosa.load(file_path, sr=sr)

                total_duration = len(y) / sr
                intervals = librosa.effects.split(y, top_db=top_db)

                silence_type = []
                max_silence = 0

                if len(intervals) == 0:

                    max_silence = total_duration
                    silence_type.append("Full")

                else:

                    if intervals[0][0] > 0:
                        start_silence = intervals[0][0] / sr
                        max_silence = max(max_silence, start_silence)
                        silence_type.append("Start")

                    if intervals[-1][1] < len(y):
                        end_silence = (len(y) - intervals[-1][1]) / sr
                        max_silence = max(max_silence, end_silence)
                        silence_type.append("End")

                    for i in range(len(intervals)-1):

                        gap = (intervals[i+1][0] - intervals[i][1]) / sr

                        if gap > 0:
                            max_silence = max(max_silence, gap)
                            silence_type.append("Middle")

                if max_silence >= threshold_sec:

                    records.append({
                        "Genre": genre,
                        "Stem": stem,
                        "Duration": round(total_duration,2),
                        "Max_Silence_Sec": round(max_silence,2),
                        "Silence_Location": ", ".join(silence_type),
                        "File_Path": file_path
                    })

    df = pd.DataFrame(records)
    return df


df_silence = find_long_silences(tr)


# ====================================
# QUESTIONS 4-9
# ====================================

q4 = len(df_silence)

print("\n--- Q4 ---")
print("Total files silence >=5:", q4)

q5 = len(df_silence[df_silence['Stem']=='vocals'])

print("\n--- Q5 ---")
print("Vocals silence >=5:", q5)

q6 = df_silence[df_silence['Stem']=='vocals']['Max_Silence_Sec'].mean()

print("\n--- Q6 ---")
print("Average silence vocals:", q6)

q7 = len(df_silence[(df_silence['Genre']=='jazz') &
                    (df_silence['Stem']=='drums')])

print("\n--- Q7 ---")
print("Jazz drums silence >=5:", q7)

q8 = len(df_silence[(df_silence['Genre']=='jazz') &
                    (df_silence['Stem']=='drums') &
                    (df_silence['Silence_Location']=='Middle')])

print("\n--- Q8 ---")
print("Jazz drums middle only:", q8)

q9 = len(df_silence[(df_silence['Genre']=='jazz') &
                    (df_silence['Stem']=='drums') &
                    (df_silence['Max_Silence_Sec']>=10)])

print("\n--- Q9 ---")
print("Jazz drums silence >=10:", q9)


# LOGGING SILENCE STATS

wandb.log({
    "files_with_silence_5s": q4,
    "vocals_silence_5s": q5,
    "avg_vocal_silence": q6,
    "jazz_drum_silence_5s": q7,
    "jazz_drum_middle_silence": q8,
    "jazz_drum_silence_10s": q9
})


# ====================================
# MIXING STEMS
# ====================================

rock_songs = sorted(os.listdir(os.path.join(DATA_ROOT,'rock')))
first_song = rock_songs[0]

stems_audio = []

for stem_file in STEMS:

    path = os.path.join(DATA_ROOT,'rock',first_song,stem_file)

    y,_ = librosa.load(path, sr=SR, duration=5.0)

    stems_audio.append(y)

stems_stack = np.vstack(stems_audio)
mix_raw = np.sum(stems_stack, axis=0)

rms_val = np.sqrt(np.mean(mix_raw**2))
max_val = np.max(np.abs(mix_raw))


print("\n--- Q10 ---")
print("Mix length:", len(mix_raw))

print("\n--- Q11 ---")
print("RMS:", round(rms_val,2))

print("\n--- Q12 ---")
print("Max peak before norm:", max_val)


# LOG MIX METRICS

wandb.log({
    "mix_length": len(mix_raw),
    "mix_rms": rms_val,
    "mix_max_peak": max_val
})


# =========================
# CELL 3 — FINISH RUN
# =========================

run.finish()
