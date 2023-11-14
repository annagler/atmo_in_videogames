from pydub import AudioSegment
import pandas as pd

def time_to_milliseconds(time):
    hh, mm, ss, ff = map(int, time.split(':'))
    return (hh * 3600 + mm * 60 + ss + ff/25) * 1000  # convert to milliseconds

def split_and_merge_audio(file_name, start_time1, end_time1, start_time2, end_time2, output_name):
    audio = AudioSegment.from_mp3(file_name)

    end_time1 = time_to_milliseconds(end_time1) - 30000  # Take the last 30 seconds
    start_time2 = time_to_milliseconds(start_time2)  # Take the first 30 seconds

    # Clip the audio segments
    clip1 = audio[end_time1:end_time1 + 30000]
    clip2 = audio[start_time2:start_time2 + 30000]

    # Merge the audio segments
    merged = clip1 + clip2
   
    # Export the new audio segment
    merged.export(output_name, format="mp3")

# Example usage:
df = pd.read_csv('../Segmentations/journey.csv')
for i in range(len(df) - 1):
    start_ts1 = df.loc[i, 'start_ts']
    end_ts1 = df.loc[i, 'end_ts']
    start_ts2 = df.loc[i+1, 'start_ts']
    end_ts2 = df.loc[i+1, 'end_ts']
    output_name = f"../Audio/Audio-Output/output_{start_ts1}_to_{end_ts1}_and_{start_ts2}_to_{end_ts2}.mp3"
    split_and_merge_audio("../Audio/Sources/journey-audio.mp3", start_ts1, end_ts1, start_ts2, end_ts2, output_name)
