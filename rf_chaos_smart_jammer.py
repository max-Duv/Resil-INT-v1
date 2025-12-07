import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# ---------------------------
# 1. Parameters
# ---------------------------
np.random.seed(0)

N_SYMS = 5000          # number of QPSK symbols
SPS    = 8             # samples per symbol (oversampling)
RRC_ROLL = 0.35
NOISE_VAR = 0.01       # baseline AWGN variance

# ---------------------------
# 2.  Root-Raised Cosine filter (RRC)
# ---------------------------
def rrc_filter(beta, span, sps):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - beta + 4*beta/np.pi
        elif abs(ti) == 1/(4*beta):
            h[i] = (beta/np.sqrt(2) *
                    ((1+2/np.pi)*np.sin(np.pi/(4*beta)) +
                     (1-2/np.pi)*np.cos(np.pi/(4*beta))))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + \
                  4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1-(4*beta*ti)**2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h**2))
    return h

# ---------------------------
# 3. QPSK Modulation
# ---------------------------
def qpsk_mod(bits):
    # group bits into 2
    bits_reshaped = bits.reshape(-1, 2)
    map_table = {
        (0,0):  1+1j,
        (0,1): -1+1j,
        (1,1): -1-1j,
        (1,0):  1-1j
    }
    syms = np.array([map_table[tuple(b)] for b in bits_reshaped])
    syms /= np.sqrt(2)  # normalize
    return syms

def qpsk_demod(syms):
    bits_out = []
    for s in syms:
        b0 = 0 if s.real >= 0 else 1
        b1 = 0 if s.imag >= 0 else 1
        # invert mapping of (b0,b1)
        if (b0,b1) == (0,0):
            bits_out.extend([0,0])
        elif (b0,b1) == (1,0):
            bits_out.extend([1,0])
        elif (b0,b1) == (1,1):
            bits_out.extend([1,1])
        else:   # (0,1)
            bits_out.extend([0,1])
    return np.array(bits_out, dtype=int)

# ---------------------------
# 4. Generate Baseband Signal
# ---------------------------
# Generating Random Bits to assign to QPSK carrier
bits = np.random.randint(0, 2, size=2*N_SYMS)

# QPSK symbols
syms = qpsk_mod(bits)

upsampled = np.zeros(N_SYMS * SPS, dtype=complex)
upsampled[::SPS] = syms

# pulse shape with RRC
rrc = rrc_filter(RRC_ROLL, span=8, sps=SPS)
tx = lfilter(rrc, 1.0, upsampled)

#  Baseline noise
noise = (np.sqrt(NOISE_VAR/2) *
         (np.random.randn(len(tx)) + 1j*np.random.randn(len(tx))))
rx_clean = tx + noise

# ---------------------------
# 5. Simple matched filter + downsample + BER
# ---------------------------
def demod_chain(rx):
    rx_filt = lfilter(rrc, 1.0, rx)
    offset = len(rrc)//2
    syms_rx = rx_filt[offset::SPS][:N_SYMS]

    bits_hat = qpsk_demod(syms_rx)
    bits_ref = bits[:len(bits_hat)]   # align lengths

    ber = np.mean(bits_hat != bits_ref)
    return ber, syms_rx


ber_clean, syms_clean = demod_chain(rx_clean)
print(f"[BASELINE] BER clean = {ber_clean:.4e}")

plt.figure()
plt.scatter(syms_clean.real, syms_clean.imag, s=3)
plt.title("Constellation - Clean")
plt.grid(True)
plt.show()


# ---------------------------
# 6. Adaptive Interference
# ---------------------------

def adaptive_interference(rx, jammer_power=0.1, fs=1.0):

    N = len(rx)
    # 1. FFT to find where energy is high
    spectrum = np.fft.fftshift(np.fft.fft(rx))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # find peak bin ("victim" center)
    peak_idx = np.argmax(np.abs(spectrum))
    f_peak = freqs[peak_idx]

    # 2. Generate narrowband tone at f_peak
    t = np.arange(N) / fs
    tone = np.exp(1j * 2*np.pi * f_peak * t)

    # normalize tone power
    tone /= np.sqrt(np.mean(np.abs(tone)**2))

    # 3. Scale tone to desired jammer power relative to rx signal
    sig_power = np.mean(np.abs(rx)**2)
    target_jam_power = jammer_power * sig_power
    tone *= np.sqrt(target_jam_power)

    # 4. Add interference
    rx_jammed = rx + tone
    return rx_jammed, f_peak

# test with one jammer power
jam_powers = [0.01, 0.1, 0.5, 1.0]
ber_list = []

for jp in jam_powers:
    rx_jammed, f_peak = adaptive_interference(rx_clean, jammer_power=jp, fs=1.0)
    ber_jam, syms_jam = demod_chain(rx_jammed)
    ber_list.append(ber_jam)
    print(f"[JAMMER] power={jp:.2f}, peak_f={f_peak:.4f}, BER={ber_jam:.4e}")

# BER vs Adaptive Jammers' Power plot
plt.figure()
plt.semilogy(jam_powers, ber_list, marker='o')
plt.xlabel("Jam power (relative)")
plt.ylabel("BER")
plt.title("BER vs Adaptive Jammer Power")
plt.grid(True, which='both')
plt.show()

# Constellation Plot
plt.figure()
plt.scatter(syms_jam.real, syms_jam.imag, s=3)
plt.title(f"Constellation - Jammed (power={jp})")
plt.grid(True)
plt.show()

def run_chaos_scenario():
    jam_seq = np.linspace(0.0, 1.0, 11)  # 0 → 1 in 0.1 steps
    results = []
    for jp in jam_seq:
        rx_jammed, _ = adaptive_interference(rx_clean, jammer_power=jp, fs=1.0)
        ber, _ = demod_chain(rx_jammed)

        # SNR estimate (linear)
        snr_est = np.mean(np.abs(tx)**2) / np.mean(np.abs(rx_jammed - tx)**2)

        # store (jam_power, ber, snr_linear)
        results.append((jp, ber, snr_est))

        print(f"J={jp:.2f} BER={ber:.3e} SNR≈{10*np.log10(snr_est):.1f} dB")

    return results

results = run_chaos_scenario()
print(f"[DEBUG] chaos results count = {len(results)}")

import csv, time
import numpy as np

csv_path = "smart_jammer_results.csv"

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # header row
    w.writerow(["timestamp_ms", "jam_power", "ber", "snr_linear", "snr_db"])
    now_ms = int(time.time() * 1000)

    for jp, ber, snr_lin in results:
        snr_db = 10 * np.log10(snr_lin + 1e-12)
        w.writerow([now_ms, jp, ber, snr_lin, snr_db])

print(f"[+] Wrote {len(results)} rows to {csv_path}")



# after computing rx_jammed (complex baseband, |x| <= 1): scale and send to PlutoSDR:

import adi

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = 2_000_000
sdr.tx_lo = 500_315_000
sdr.tx_hardwaregain_chan0 = -10  # dB

# normalize amplitude
iq_tx = rx_jammed / np.max(np.abs(rx_jammed)) * 0.8
sdr.tx(iq_tx.astype(np.complex64))
