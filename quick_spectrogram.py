import numpy as np
import matplotlib.pyplot as plt # type: ignore

def do_quick(fname: str, sample_rate_hz: float, ref_level_db: float, window_size: int, num_windows: int, start_crop = None, end_crop = None):

    data = np.fromfile(fname, dtype=np.float32)
    x = data.view(np.complex64)[start_crop:end_crop]

    print(f"File size: {len(data) * 4 / (1 << 20):.3f} MB // {len(x) / 1e6:.3f} MSamples = {len(x) / sample_rate_hz:.3f} s @ {sample_rate_hz / 1e6:.3f} MHz")

    title = "..." + fname[-25:]

    plot_spectrogram(x, sample_rate_hz, ref_level_db, window_size, num_windows, title)

def plot_spectrogram(x, sample_rate_hz: float, ref_level_db: float, window_size: int, num_windows: int, title: str = "", do_plot: bool = True, verbose=True, red=False):

    W = window_size
    K = num_windows

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot average of every N samples
    N = 1000
    
    drop = len(x) % N
    if drop == 0:
        data_to_average = x
    else:
        data_to_average = x[:-drop]

    if verbose:
        print(f" * Plotting average of every {N} samples --> {len(data_to_average) / N} points representing {N / sample_rate_hz * 1e3:.3f} ms each")
    average = np.average(np.reshape(20 * np.log10(np.abs(data_to_average)) + ref_level_db, [-1, N]), axis=1)
    times = np.linspace(0, ((len(data_to_average) - 1) / sample_rate_hz), round(len(data_to_average) / N))
    color = "C3" if red else None
    ax1.plot(times * 1e3, average, linewidth=1, color=color)
    ax1.set_title(title)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Power (dB)")

    # Plot short-time Fourier transform (selected windows - NOT necessarily the whole things)
    df = sample_rate_hz / W
    
    interval_size = int(np.floor(len(x) / K))
    ignored_fraction = (interval_size - W) / interval_size

    if verbose:
        print(f" * Plotting STFT with windows length {W} ({W / sample_rate_hz * 1e3:.3f} ms, df = {df / 1e6:.3f} MHz)")
        print(f"     NOTE: only plotting {K:,} windows which means {ignored_fraction:.2%} of data is ignored,\n" +
            f"     i.e. {ignored_fraction * interval_size / sample_rate_hz * 1e3:.3f} ms every {interval_size / sample_rate_hz * 1e3:.3f} ms")
    
    drop  = len(x) - interval_size * K
    if drop > 0:
        if verbose:
            print(f"     > INFO: ignoring last {drop} samples, since {K} is not a factor of {len(x)}")
        x = x[:-drop]

    data_reshaped = np.reshape(x, [-1, interval_size])[:, :W] # select window-sized data from each interval

    spectrogram = 20 * np.log10(
        np.abs(np.fft.fftshift(
            np.fft.fft(data_reshaped, axis=1), axes=[1
            ]
        ))
    )

    im = ax2.imshow(spectrogram + ref_level_db, aspect='auto', extent = [
        - sample_rate_hz / 2 / 1e6,
        (sample_rate_hz / 2 - df) / 1e6,
        (len(x) - drop) / sample_rate_hz * 1e3, 0
    ])

    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Time (ms)")
    ax2.invert_yaxis()
    colorbar = plt.colorbar(im, ax=ax2)
    colorbar.set_label("Power (dB)")

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    if do_plot:
        plt.show()
    else:
        return ax1, ax2

def main():
    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_170515_107300000_1800000_fc.raw"
    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_202931_111000000_1800000_fc.raw"
    fname = "/home/jpiland/iq_recordings/gqrx_20251206_205810_111000000_1800000_fc.raw"
    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_203920_111350000_1800000_fc.raw"
    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_211543_111000000_1800000_fc.raw"
    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_210437_109900000_1800000_fc.raw"

    Fs = 1.8e6

    do_quick(
        fname=fname, 
        sample_rate_hz=Fs, 
        ref_level_db=0, 
        window_size=1024 * 16, 
        num_windows=1024 // 8,
        # start_crop=int(10 * Fs),
        # end_crop=int(11 * Fs),
        # start_crop=int(24 * Fs),
        # end_crop=int(25 * Fs),
    )

if __name__ == "__main__":
    main()

