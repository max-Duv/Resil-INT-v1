import time
import numpy as np
import adi
from prometheus_client import start_http_server, Gauge, Counter

sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(915e6)
sdr.sample_rate = int(1e6)
sdr.rx_rf_bandwidth = int(500e3)
sdr.rx_buffer_size = 1024*8

g_snr_db = Gauge("resilint_snr_db", "Estimated SNR in dB")
g_lock   = Gauge("resilint_lock_state", "1 if sync locked, 0 otherwise")
c_frames = Counter("resilint_frames_total", "Total frames observed")
c_frames_ok = Counter("resilint_frames_ok", "Frames decoded successfully")
g_ber    = Gauge("resilint_ber", "Bit error rate (moving estimate)")

start_http_server(8000)

ber_est = 0.0

def estimate_snr(iq):
    p_sig = np.mean(np.abs(iq)**2)
    noise = np.percentile(np.abs(iq)**2, 10)
    snr_lin = (p_sig - noise) / (noise + 1e-12)
    return 10 * np.log10(max(snr_lin, 1e-6))

def fake_frame_decode(iq):
    snr = estimate_snr(iq)
    locked = snr > 5
    ber = float(np.clip(0.5 * np.exp(-snr/5), 1e-6, 0.5))
    success = locked and (ber < 0.1)
    return snr, locked, success, ber

while True:
    iq = sdr.rx()
    snr, locked, success, ber = fake_frame_decode(iq)
    c_frames.inc()
    if success:
        c_frames_ok.inc()

    # simple EMA on BER
    ber_est = 0.9 * ber_est + 0.1 * ber if c_frames._value.get() > 1 else ber

    g_snr_db.set(snr)
    g_lock.set(1 if locked else 0)
    g_ber.set(ber_est)

    time.sleep(0.2)
