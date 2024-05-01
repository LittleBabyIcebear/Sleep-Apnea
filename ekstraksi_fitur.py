#Import packages
import biosppy
import pyhrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
import numpy as np
import pandas as pd

signal = np.loadtxt('C:\\Database\\Sleep Aja Ga Sih!\\Dataset\\After Cut\\Normal\\a15_2.csv')[:]

#t peaks
t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal, sampling_rate=100)[:3]

def compute_feature(t, rpeaks):
    #hr_mean 
    results_hrv = td.hr_parameters(rpeaks=t[rpeaks])["hr_mean"]
    df_hr_mean = pd.DataFrame({'HR': [results_hrv]})

    #SDNN
    sdnn_value = td.sdnn(rpeaks=t[rpeaks])["sdnn"]
    df_sdnn = pd.DataFrame({'SDNN': [sdnn_value]})

    #RMSDD 
    rmssd_value = td.rmssd(rpeaks=t[rpeaks])["rmssd"]
    df_rmssd = pd.DataFrame({'RMSSD': [rmssd_value]})

    #SDSD 
    sdsd_value = td.sdsd(rpeaks=t[rpeaks])["sdsd"]
    df_sdsd = pd.DataFrame({'SDSD': [sdsd_value]})

    ##pNN50 
    pnn50_value = td.nn50(rpeaks=t[rpeaks])["pnn50"]
    df_pnn50 = pd.DataFrame({'pNN50': [pnn50_value]})

    #LF/HF 
    LFHF_Ratio = fd.welch_psd(rpeaks=t[rpeaks])["fft_ratio"]
    df_LFHF_Ratio = pd.DataFrame({'LF/HF': [LFHF_Ratio]})

    #LF #HF 
    lf_hf=fd.welch_psd(rpeaks=t[rpeaks], show=False)["fft_peak"]
    LF = lf_hf[1]
    HF = lf_hf[2]
    df_lf = pd.DataFrame({'LF': [LF]})
    df_hf = pd.DataFrame({'HF': [HF]})

    #sd1 #sd2
    sd1_sd2 = nl.poincare(rpeaks=t[rpeaks])
    sd1_value = sd1_sd2['sd1']
    sd2_value = sd1_sd2['sd2']
    sd_ratio_value = sd1_sd2['sd_ratio']
    df_sd1 = pd.DataFrame({'SD1': [sd1_value]})
    df_sd2 = pd.DataFrame({'SD2': [sd2_value]})
    df_sd_ratio = pd.DataFrame({'SD_ratio': [sd_ratio_value]})

    #Combine all 
    df_result = pd.concat([df_hr_mean, 
                        df_sdnn, 
                        df_rmssd, 
                        df_sdsd, 
                        df_pnn50,
                        df_lf, 
                        df_hf,
                        df_LFHF_Ratio, 
                        df_sd1, 
                        df_sd2, 
                        df_sd_ratio], axis=1)
    df_all=pd.DataFrame(df_result)

    return df_all

df = compute_feature(t, rpeaks)
