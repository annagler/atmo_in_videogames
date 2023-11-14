#!/usr/bin/env python
# coding: utf-8

# In[34]:


#SEGMENTATIONTempo


# In[35]:


#Calculate Segments via Kernel Change Point Detection using Tempo Information


# In[36]:


import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Audio
import ruptures as rpt
import pandas as pd


# In[37]:


#name="../Output/journey/journey_"
#path = "../" +"Data/journey/Journey - Gameplay  Playthrough (No Commentary)-[AudioTrimmer.com].mp3"

name="../Output/mario/mario_"
path = "../" +"Data/mario/Super Mario 64 [Part 1꞉ Bob-omb Battlefield] (No Commentary)-[AudioTrimmer.com].mp3"

#name="../Output/silenthill/silenthill_"
#path = "../" +"Data/silenthill/Silent Hill  Full UHD 4K  Longplay Walkthrough Gameplay No Commentary - 01.mp3"


x, Fs = librosa.load(path)
    
if len(x.shape) != 1:
    x = 0.5+(x[:, 0]+x[:, 1])

Audio(x, rate=Fs)

length_in_seconds = x.shape[0]/Fs
print('Sample rate: Fs=%0.0f Hz' % (Fs))
print('Length: %1d:%02d minutes' % (length_in_seconds//60, np.remainder(length_in_seconds, 60)))


# In[38]:


def fig_ax(figsize=(15, 5), dpi=150):
    """Return a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)


# In[39]:


# Compute the onset strength
hop_length_tempo = 256
oenv = librosa.onset.onset_strength(
    y=x, sr=Fs, hop_length=hop_length_tempo
)

# Compute the tempogram
tempogram = librosa.feature.tempogram(
    onset_envelope=oenv,
    sr=Fs,
    hop_length=hop_length_tempo,
)
# Display the tempogram
fig, ax = fig_ax()
_ = librosa.display.specshow(
    tempogram,
    ax=ax,
    hop_length=hop_length_tempo,
    sr=Fs,
    x_axis="s",
    y_axis="tempo",
)


# In[40]:


# Choose detection method
algo = rpt.KernelCPD(kernel="linear").fit(tempogram.T)

# Choose the number of changes (elbow heuristic)
n_bkps_max = 10  # K_max
# Start by computing the segmentation with most changes.
# After start, all segmentations with 1, 2,..., K_max-1 changes are also available for free.
_ = algo.predict(n_bkps_max)

array_of_n_bkps = np.arange(1, n_bkps_max + 1)


def get_sum_of_cost(algo, n_bkps) -> float:
    """Return the sum of costs for the change points `bkps`"""
    bkps = algo.predict(n_bkps=n_bkps)
    return algo.cost.sum_of_costs(bkps)


fig, ax = fig_ax((7, 4))
ax.plot(
    array_of_n_bkps,
    [get_sum_of_cost(algo=algo, n_bkps=n_bkps) for n_bkps in array_of_n_bkps],
    "-*",
    alpha=0.5,
)
ax.set_xticks(array_of_n_bkps)
ax.set_xlabel("Number of change points")
ax.set_title("Sum of costs")
ax.grid(axis="x")
ax.set_xlim(0, n_bkps_max + 1)

# Visually we choose n_bkps=5 (highlighted in red on the elbow plot)
n_bkps = 5
_ = ax.scatter([5], [get_sum_of_cost(algo=algo, n_bkps=5)], color="r", s=100)


# In[41]:


# Segmentation
bkps = algo.predict(n_bkps=n_bkps)
# Convert the estimated change points (frame counts) to actual timestamps
bkps_times = librosa.frames_to_time(bkps, sr=Fs, hop_length=hop_length_tempo)

# Displaying results
fig, ax = fig_ax()
_ = librosa.display.specshow(
    tempogram,
    ax=ax,
    x_axis="s",
    y_axis="tempo",
    hop_length=hop_length_tempo,
    sr=Fs,
)

for b in bkps_times[:-1]:
    ax.axvline(b, ls="--", color="white", lw=4)
    
plt.savefig(name + "tempoSegments.png")


# In[42]:


# Compute change points corresponding indexes in original signal
bkps_time_indexes = (Fs * bkps_times).astype(int).tolist()

for segment_number, (start, end) in enumerate(
    rpt.utils.pairwise([0] + bkps_time_indexes), start=1
):
    segment = x[start:end]
    print(f"Segment n°{segment_number} (duration: {segment.size/Fs:.2f} s)")
    display(Audio(data=segment, rate=Fs))


# In[43]:


# Convert seconds to "HH:MM:SS:MS"-format
def convert_to_time_format(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 100)
    return "{:02}:{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds), milliseconds)

# Convert "HH:MM:SS:MS"-format to seconds
def convert_to_seconds(timestring):
    hours, minutes, seconds, milliseconds = map(int, timestring.split(':'))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 100


# Save data as DataFrame
df = pd.DataFrame(bkps_times, columns=["end_ts"])

# Add new columns
df = df.assign(segment_id=np.arange(1, len(df) + 1),
               segmentation_name="audio-tempo-segmentation")

# Rename the "start_ts" column to "end_ts" and apply conversion function
df["end_ts"] = df["end_ts"].apply(convert_to_time_format)

# Create the "start_ts" column
df["start_ts"] = df["end_ts"].shift(fill_value=convert_to_time_format(0))

# Create the "duration_ts" column
df["duration_ts"] = (df["end_ts"].apply(convert_to_seconds) - df["start_ts"].apply(convert_to_seconds)).apply(convert_to_time_format)

# Rearrange columns
df = df[["segment_id", "segmentation_name", "start_ts", "end_ts", "duration_ts"]]

# Save DataFrame to a CSV file
df.to_csv(name + "bkps_timesTempo.csv", index=True)


# In[ ]:




