import numpy as np
from scipy.fft import fft


def dirac(x):
    if x == 0:
        dirac_delta = 1
    else:
        dirac_delta = 0
    return dirac_delta


def countRespiratoryRate(jumlahdata, y):
    fs = 125
    M = 20  # lebar window
    w = np.zeros(M)
    for n in range(M - 1):
        w[n] = 0.54 - 0.46 * np.cos((2 * n * np.pi) / M)

    delay3 = 3
    gradien3 = np.zeros(jumlahdata)
    qj = np.zeros((6, 100000))
    a = -(round(2**3) + round(2 ** (3 - 1)) - 2)
    b = -(1 - round(2 ** (3 - 1)))

    for k in range(-10, 4):
        qj[3, k + abs(a)] = (
            -1
            / 32
            * (
                dirac(k - 3)
                + 3 * dirac(k - 2)
                + 6 * dirac(k - 1)
                + 10 * dirac(k)
                + 11 * dirac(k + 1)
                + 9 * dirac(k + 2)
                + 4 * dirac(k + 3)
                - 4 * dirac(k + 4)
                - 9 * dirac(k + 5)
                - 11 * dirac(k + 6)
                - 10 * dirac(k + 7)
                - 6 * dirac(k + 8)
                - 3 * dirac(k + 9)
                - dirac(k + 10)
            )
        )

    w2fb = np.zeros((6, jumlahdata + 15))
    for n in range(jumlahdata):
        w2fb[3][n + 3] = 0
        for k in range(a, b + 1):
            w2fb[3][n + 3] += qj[3, (k + abs(a))] * y[n - (k + abs(a))]

    for n in range(delay3, jumlahdata):
        gradien3[n] = w2fb[3][n - delay3] - w2fb[3][n + delay3]

    hasilQRS = np.zeros(jumlahdata)
    for n in range(jumlahdata):
        if gradien3[n] > 1.5:
            hasilQRS[n - 8] = 5
        else:
            hasilQRS[n - 8] = 0

    ptp = 0
    waktu = np.zeros(np.size(hasilQRS))
    selisih = np.zeros(np.size(hasilQRS))
    for n in range(np.size(hasilQRS) - 1):
        if hasilQRS[n] < hasilQRS[n + 1]:
            waktu[ptp] = n / fs
            selisih[ptp] = waktu[ptp] - waktu[ptp - 1]
            ptp += 1

    ptp = ptp - 1
    if ptp < 1:
        return 0

    j = 0
    peak = np.zeros(np.size(hasilQRS))
    for n in range(np.size(hasilQRS)):
        if (hasilQRS[n] == 5) and (hasilQRS[n - 1] == 0):
            peak[j] = n
            j += 1

    temp = 0
    interval = np.zeros(np.size(hasilQRS))
    BPM = np.zeros(np.size(hasilQRS))
    for n in range(ptp):
        interval[n] = (peak[n] - peak[n - 1]) / fs
        BPM[n] = 60 / interval[n]
        temp = temp + BPM[n]
        rata = temp / (n - 1)

    bpm_rr = np.zeros(ptp)
    rr_t = np.zeros(ptp)
    for n in range(ptp):
        bpm_rr[n] = 60 / selisih[n]
        rr_t[n] = selisih[n]
        if bpm_rr[n] > 100:
            bpm_rr[n] = rata

    # Calculate mean RR interval
    mean_rr = np.mean(rr_t)

    # Calculate fs_hrv
    fs_hrv = 1 / mean_rr

    bpm_rrn = bpm_rr - (np.sum(bpm_rr) / len(bpm_rr))

    array_bpm = np.zeros(ptp)
    sinyal_bpm = np.zeros(ptp)
    nn = 0
    while True:
        window_ptp = np.zeros(ptp)
        jj = 0
        for i in range(round((M / 2) * nn), round(((M / 2) * nn)) + M):
            if i >= ptp:
                continue
            else:
                window_ptp[i] = w[jj]
                jj += 1

        for i in range(ptp):
            sinyal_bpm[i] = bpm_rrn[i] * window_ptp[i]

        array_bpm1 = fft(sinyal_bpm)
        array_bpm2 = np.abs(array_bpm1)
        for i in range(ptp):
            array_bpm[i] += array_bpm2[i]

        nn += 1

        if round((M / 2) * nn) > ptp:
            break

    array_bpm = array_bpm / nn

    HF_PSD = np.zeros(ptp // 2)
    for i in range(ptp // 2):
        if i * fs_hrv / ptp > 0.15 and i * fs_hrv / ptp < 0.4:
            HF_PSD[i] = array_bpm[i]

    MPFx = 0
    for i in range(ptp // 2):
        MPFx += (i * fs_hrv / ptp) * HF_PSD[i]
    MPFy = np.sum(HF_PSD)
    if MPFy == 0:
        return 0

    MPF = MPFx / MPFy
    RRT = MPF * 60
    return RRT
