import pandas as pd
import datetime

# CSV-Dateien laden
df1 = pd.read_csv('segmentation.csv')
df2 = pd.read_csv('colorimetryExport.csv')

# Zeiten in datetime-Objekten konvertieren
df1['start_ts'] = pd.to_datetime(df1['start_ts'], format='%H:%M:%S:%f').dt.time
df1['end_ts'] = pd.to_datetime(df1['end_ts'], format='%H:%M:%S:%f').dt.time
df2['time_s'] = df2['time_ms'] / 1000


# Durchschnittliche luminance und chroma für ein gegebenes Zeitintervall berechnen
def calculate_average_values(time_start, time_end):
    time_start_seconds = time_start.hour * 3600 + time_start.minute * 60 + time_start.second + time_start.microsecond / 1E6
    time_end_seconds = time_end.hour * 3600 + time_end.minute * 60 + time_end.second + time_end.microsecond / 1E6
    mask = (df2['time_s'] >= time_start_seconds) & (df2['time_s'] <= time_end_seconds)
    average_luminance = df2.loc[mask, 'luminance'].mean()
    average_chroma = df2.loc[mask, 'chroma'].mean()
    return average_luminance, average_chroma


# Durch jedes Segment gehen und die durchschnittlichen Werte berechnen
for i in range(1, len(df1)):
    segment_end = df1.loc[i - 1, 'end_ts']
    segment_start = df1.loc[i, 'start_ts']

    # Erstellt zwei datetime.time Objekte, die 10 Sekunden vor dem Ende des aktuellen Segments und 10 Sekunden nach dem Start des nächsten Segments liegen
    segment_end_minus_10s = (
                datetime.datetime.combine(datetime.date.today(), segment_end) - datetime.timedelta(seconds=10)).time()
    segment_start_plus_10s = (
                datetime.datetime.combine(datetime.date.today(), segment_start) + datetime.timedelta(seconds=10)).time()

    # Berechnet die durchschnittlichen Werte für diese beiden Zeitintervalle
    avg_luminance_end, avg_chroma_end = calculate_average_values(segment_end_minus_10s, segment_end)
    avg_luminance_start, avg_chroma_start = calculate_average_values(segment_start, segment_start_plus_10s)

    print(f'Für den Übergang von Segment {df1.loc[i - 1, "segment_id"]} zu Segment {df1.loc[i, "segment_id"]}:')
    print(f'Durchschnittliche luminance am Ende des vorherigen Segments: {avg_luminance_end}, Durchschnittliche chroma am Ende des vorherigen Segments: {avg_chroma_end}')
    print(f'Durchschnittliche luminance am Anfang des aktuellen Segments: {avg_luminance_start}, Durchschnittliche chroma am Anfang des aktuellen Segments: {avg_chroma_start}\n')
