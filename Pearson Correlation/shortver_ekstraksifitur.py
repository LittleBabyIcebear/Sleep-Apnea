import os
import numpy as np
import pandas as pd
import biosppy
import pyhrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# set plt ke nonaktif
plt.ioff()

# Fungsi untuk menghitung fitur dari satu file
def compute_feature(file_path):
    print("Mulai menghitung fitur untuk file:", file_path)
    signal = np.loadtxt(file_path)[:]

    #t peaks
    t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal, sampling_rate=100)[:3]

    # Hitung fitur HRV
    results_hrv = td.hr_parameters(rpeaks=t[rpeaks])["hr_mean"]
    sdnn_value = td.sdnn(rpeaks=t[rpeaks])["sdnn"]
    rmssd_value = td.rmssd(rpeaks=t[rpeaks])["rmssd"]
    sdsd_value = td.sdsd(rpeaks=t[rpeaks])["sdsd"]
    pnn50_value = td.nn50(rpeaks=t[rpeaks])["pnn50"]
    LFHF_Ratio = fd.welch_psd(rpeaks=t[rpeaks])["fft_ratio"]
    lf_hf=fd.welch_psd(rpeaks=t[rpeaks], show=False)["fft_peak"]
    LF = lf_hf[1]
    HF = lf_hf[2]
    sd1_sd2 = nl.poincare(rpeaks=t[rpeaks])
    sd1_value = sd1_sd2['sd1']
    sd2_value = sd1_sd2['sd2']
    sd_ratio_value = sd1_sd2['sd_ratio']

    # Buat dataframe
    df_result = pd.DataFrame({
        'HR': [results_hrv],
        'SDNN': [sdnn_value],
        'RMSSD': [rmssd_value],
        'SDSD': [sdsd_value],
        'pNN50': [pnn50_value],
        'LF': [LF],
        'HF': [HF],
        'LF/HF': [LFHF_Ratio],
        'SD1': [sd1_value],
        'SD2': [sd2_value],
        'SD_ratio': [sd_ratio_value]
    })
    print("Selesai menghitung fitur untuk file:", file_path)

    return df_result

# Fungsi untuk menghasilkan dataframe dari setiap file dalam direktori
def process_directory(directory_path):
    df_list = []

    # Iterasi setiap file dalam direktori
    for filename in os.listdir(directory_path):
        if filename.endswith((".csv", ".txt")):
            file_path = os.path.join(directory_path, filename)
            df = compute_feature(file_path)
            df_list.append(df)

    # Gabungkan dataframe dari setiap file menjadi satu dataframe tunggal
    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)
        return df_combined
    else:
        return None

# Direktori yang berisi file-file yang ingin diproses
directory_path = "Apnea_v2"

# Proses direktori
df_combined = process_directory(directory_path)

# Simpan dataframe ke dalam file Excel
if df_combined is not None:
    excel_output_path = "fitur_apnea_v2.xlsx"
    df_combined.to_excel(excel_output_path, index=False)
    print("Dataframe berhasil disimpan ke dalam file Excel:", excel_output_path)
else:
    print("Tidak ada file yang diproses dalam direktori:", directory_path)
