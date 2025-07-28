import pandas as pd
from scipy.signal import savgol_filter


df = pd.read_csv("sv.csv")

filter_window=1000

df['Sv_tot (cm)'] = savgol_filter(df['Sv_tot (cm)'], filter_window, 3)
df['Sv_trn (cm)'] = savgol_filter(df['Sv_trn (cm)'], filter_window, 3)
df['Sv_rot (cm)'] = savgol_filter(df['Sv_rot (cm)'], filter_window, 3)
df['Sv_vib (cm)'] = savgol_filter(df['Sv_vib (cm)'], filter_window, 3)
df['Sv_trn_s (cm)'] = savgol_filter(df['Sv_trn_s (cm)'], filter_window, 3)
df['Sv_trn_g (cm)'] = savgol_filter(df['Sv_trn_g (cm)'], filter_window, 3)
df['Sv_rot_s (cm)'] = savgol_filter(df['Sv_rot_s (cm)'], filter_window, 3)
df['Sv_rot_g (cm)'] = savgol_filter(df['Sv_rot_g (cm)'], filter_window, 3)

df.to_csv('sv_smoothed.csv', index=False)
