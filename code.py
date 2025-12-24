import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def step1_data_analysis(in_w, out_w, out_m, h_des_p):
    files = [in_w, out_w, out_m]
    names = ["Input White Noise", "Output White Noise", "Output Music"]
    
    for name, path in zip(names, files):
        data, fs = sf.read(path)
        print(f"{name}:")
        print(f"  Type: {data.dtype}, Shape: {data.shape}, Length: {len(data)}")
        print(f"  Rate: {fs} Hz, Duration: {len(data)/fs:.4f} s")

    H_des = np.load(h_des_p)
    fs = 44100
    f = np.linspace(0, fs/2, len(H_des))
    h_des = np.fft.irfft(H_des)
    delay = np.argmax(np.abs(h_des))
    
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.semilogx(f, 20 * np.log10(np.abs(H_des) + 1e-10))
    plt.title("H_des Magnitude Response")
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency - Log Axis (Hz)")
    plt.subplot(3, 1, 2)
    plt.plot(f, np.unwrap(np.angle(H_des)))
    plt.title("H_des Phase")
    plt.ylabel("Phase (Radians)")
    plt.xlabel("Frequency - Linear Axis (Hz)")
    plt.subplot(3, 1, 3)
    plt.plot(h_des)
    plt.title(f"h_des IRF - Peak Delay: {delay} samples")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()
    
    return H_des, fs, h_des

def step2_system_id(in_w, out_w, fs):
    x_white, _ = sf.read(in_w)
    y_white, _ = sf.read(out_w)
    
    corr = signal.correlate(y_white, x_white, mode='valid')
    best_lag = np.argmax(corr)
    print(f"System ID: Detected Lag = {best_lag}")
    
    y_aligned = y_white[best_lag : best_lag + len(x_white)]
    
    f, t, Z_xx = signal.stft(x_white, fs=fs, nperseg=8192)
    _, _, Z_yy = signal.stft(y_aligned, fs=fs, nperseg=8192)
    
    P_xx = np.mean(np.abs(Z_xx)**2, axis=1)
    P_xy = np.mean(Z_yy * np.conj(Z_xx), axis=1)
    H_est = P_xy / (P_xx + 1e-15)
    
    h_est = np.fft.irfft(H_est)
    delay_est = np.argmax(np.abs(h_est))
    
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.semilogx(f, 20 * np.log10(np.abs(H_est) + 1e-10))
    plt.title("H_est Magnitude")
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency - Log Axis (Hz)")
    plt.subplot(3, 1, 2)
    plt.plot(f, np.unwrap(np.angle(H_est)))
    plt.title("H_est Phase")
    plt.ylabel("Phase (Radians)")
    plt.xlabel("Frequency - Linear Axis (Hz)")
    plt.subplot(3, 1, 3)
    plt.plot(h_est)
    plt.title(f"h_est IRF - Peak Delay after alignment: {delay_est}")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()

    # energy = np.cumsum(h_est**2) / np.sum(h_est**2)
    # plt.plot(energy)
    # plt.title("Cumulative Energy Concentration")
    # plt.xlabel("Samples")
    # plt.ylabel("Normalized Energy")
    # plt.show()
    
    return H_est, x_white, y_white, best_lag

def step3_design_compensator(H_des, H_est):
    # f = np.linspace(0, fs/2, len(H_est))

    # beta = np.full_like(f, 1e-4) 
   
    # beta[f < 100] = 1e-3 
    
    # beta[f > 18000] = 1e-2
    
    # H_est_mag_sq = np.abs(H_est)**2
    # H_comp = (H_des * np.conj(H_est)) / (H_est_mag_sq + beta)
    # H_est_mag_sq = np.abs(H_est)**2
    # H_comp = (H_des * np.conj(H_est)) / (H_est_mag_sq + 1e-4)
    H_comp = H_des/H_est
    # mag_ratio = np.abs(H_des) / (np.abs(H_est) + 1e-10)
    # phase_target = np.angle(H_des)
    # H_comp = mag_ratio * np.exp(1j * phase_target)
    return H_comp

def step4_validation(fs, x_white, y_white, H_des, H_comp, H_est):
    f = np.linspace(0, fs/2, len(H_des))
    H_est_comp = H_est * H_comp
    
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.semilogx(f, 20 * np.log10(np.abs(H_est) + 1e-10), label='H_est', alpha=0.5)
    plt.semilogx(f, 20 * np.log10(np.abs(H_est_comp) + 1e-10), label='H_est_comp', linewidth=2)
    plt.semilogx(f, 20 * np.log10(np.abs(H_des) + 1e-10), 'r--', label='H_des', alpha=0.7)
    plt.title("Magnitude Response Comparison (Linear Frequency)")
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency - Log Axis (Hz)")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(f, np.unwrap(np.angle(H_est)), label='H_est', alpha=0.5)
    plt.plot(f, np.unwrap(np.angle(H_est_comp)), label='H_est_comp', linewidth=2)
    plt.plot(f, np.unwrap(np.angle(H_des)), 'r--', label='H_des', alpha=0.7)
    plt.title("Phase Response Comparison (Linear Axis)")
    plt.ylabel("Phase (Radians)")
    plt.xlabel("Frequency - Linear Axis (Hz)")
    plt.legend()
    plt.subplot(3, 1, 3)
    h_est = np.fft.irfft(H_est)
    h_des = np.fft.irfft(H_des)
    h_est_comp = np.fft.irfft(H_est_comp)
    plt.plot(h_est, label='h_est', alpha=0.5)
    plt.plot(h_est_comp, label='h_est_comp', linewidth=2)
    plt.plot(h_des, 'r--', label='h_des', alpha=0.7)
    plt.title("IRF Comparison")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    h_comp = np.fft.irfft(H_comp)
    y_target = fftconvolve(x_white, h_des, mode='full')
    y_same = y_white[128 : 128 + len(x_white)]
    y_compensated = fftconvolve(y_same, h_comp, mode='full')
    mse = np.mean((y_target - y_compensated)**2)
    print(f"MSE Analysis: Target Len: {len(y_target)}, Comp Len: {len(y_compensated)}")
    print(f"MSE = {mse:.10}")
    
    error_spectrum_analysis(y_target, y_compensated, fs)
    
    return h_comp, h_est_comp

def error_spectrum_analysis(y_target, y_compensated, fs):
    error_signal = y_target - y_compensated
    f, P_error = signal.welch(error_signal, fs, nperseg=4096)
    _, P_target = signal.welch(y_target, fs, nperseg=4096)
    
    plt.figure(figsize=(8, 6))
    plt.semilogx(f, 10*np.log10(P_target + 1e-12), label='Target Signal Spectrum')
    plt.semilogx(f, 10*np.log10(P_error + 1e-12), label='Error (Residual) Spectrum', alpha=0.8)
    plt.fill_between(f, -100, 10*np.log10(P_error + 1e-12), color='orange', alpha=0.2)
    plt.title("Residual Spectrum Analysis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def pre_ringing_check(h_est_comp, h_des):
    plt.figure(figsize=(8, 6))
    plt.plot(h_des[0:300], label='h_des', color='r', linestyle='--', linewidth=2.5)
    plt.plot(h_est_comp[0:300], label='h_est_comp', color='orange')
    plt.title("h_est_comp IRF (Check for Pre-ringing)")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.axvline(x=128, color='b', linestyle='--', label='Target Delay')
    plt.legend()
    plt.show()

def step5_output_music(out_m, h_comp, best_lag, fs):
    y_music, _ = sf.read(out_m)
    y_m_aligned = y_music[best_lag : best_lag + int(7.0 * fs)]
    y_final = fftconvolve(y_m_aligned, h_comp, mode='full')

    sf.write('output_music_compensated_110033154.wav', y_final, fs)
    print(f"Output saved. Final Duration: {len(y_final)/fs:.4f} s")

def main():
    in_w = 'data/input_white_noise.wav'
    out_w = 'data/output_white_noise.wav'
    out_m = 'data/output_music.wav'
    h_des_p = 'data/desired FRF.npy'
    
    H_des, fs, h_des = step1_data_analysis(in_w, out_w, out_m, h_des_p)
    H_est, x_white, y_white, best_lag = step2_system_id(in_w, out_w, fs)
    H_comp = step3_design_compensator(H_des, H_est)
    h_comp, h_est_comp = step4_validation(fs, x_white, y_white, H_des, H_comp, H_est)
    pre_ringing_check(h_est_comp, h_des)
    step5_output_music(out_m, h_comp, best_lag, fs)

if __name__ == "__main__":
    main()