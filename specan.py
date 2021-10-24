import numpy as np
import scipy.io.wavfile
import scipy.stats


np.seterr(divide='ignore')


def fund(wave, fs=8000, wl=512, fmax=4000, threshold=None, ovlp=0):
    """Estimates the fundamental frequency through a short-term
       cepstral transform as per R seewave library.

    Args:
        wave (array): WAV array returned by scipy.io wavfile.read.
        fs (int, optional): sampling frequency in HZ. Defaults to 8000.
        wl (int, optional): length of window. Defaults to 512.
        fmax (int, optional): the maximum frequency to detect (in Hz).
            Defaults to 4000.
        threshold (int, optional): amplitude threshold for signal detection
            (in %). Defaults to None.
        ovlp (int, optional): overlap between two successive windows (in %).
            Defaults to 0.

    Returns:
        array:  the 1st column corresponding to time in seconds (x-axis) and
                the 2nd column corresponding to fundamental frequency in kHz (y-axis).
    """
    WL = wl // 2
    if threshold:
        cutoff = threshold / 100 * np.abs(wave).max()
        wave = np.where(np.abs(wave) < cutoff, 1e-06, wave)

    n = len(wave)
    step = np.arange(0, n+1-wl, step=int(wl - (ovlp * wl / 100)))
    N = len(step)
    z1 = np.zeros((wl, N))
    for e, i in enumerate(step):
        z1[:, e] = np.real(
            np.fft.ifft(np.log(np.abs(np.fft.fft(wave[i:(wl + i)]))))
        )
    z2 = z1[0:WL, ]
    z = np.where(np.isnan(z2) | np.isinf(z2), 0, z2)
    fmaxi = fs // fmax
    tfund = np.zeros(N)
    for i in range(N):
        tfund[i] = np.argmax(z[fmaxi:, i])
    tfund = np.where(tfund == 0, np.nan, tfund)
    ffund = fs / (tfund + fmaxi)
    x = np.linspace(0, n / fs, num=N)
    y = ffund / 1000
    return np.transpose([x, y])


def dfreq(wave, fs=8000, wl=512, wn="hanning", ovlp=0,
          bandpass=None, threshold=None, clip=None):
    """Computes the dominant frequency (i. e. the frequency of highest amplitude)
       of a time wave.

    Args:
        wave (array): WAV array returned by scipy.io wavfile.read.
        fs (int, optional): sampling frequency in HZ. Defaults to 8000.
        wl (int, optional): length of window. Defaults to 512.
        wn (str, optional): window name. Defaults to "hanning".
        ovlp (int, optional): overlap between two successive windows (in %). Defaults to 0.
        bandpass (tuple, optional): lower and upper limits of a frequency bandpass filter (in Hz). Defaults to None.
        threshold (int, optional): amplitude threshold for signal detection (in %). Defaults to None.
        clip (None, optional): dominant frequency values according to their amplitude in reference to a
            maximal value of 1 for the whole signal (has to be >0 & < 1).. Defaults to None.

    Raises:
        ValueError: if bandpass length is size other than 2.
        ValueError: if 1st element is not less than the 2nd.
        ValueError: if clip is not between 0 and 1.

    Returns:
        array: the 1st column corresponding to time in seconds (x-axis) and
               the 2nd column corresponding to to dominant frequency in kHz (y-axis).
    """
    if bandpass:
        if len(bandpass) != 2:
            raise ValueError("The argument 'bandpass' should be a tuple of length 2.")
        elif bandpass[0] > bandpass[1] or bandpass[0] == bandpass[1]:
            raise ValueError("The first element of 'bandpass' has to be less than the second.")
    if clip and not 0 < clip < 1:
        raise ValueError("'clip' has to be more than 0 and less than 1.")
    if threshold:
        cutoff = threshold / 100 * np.abs(wave).max()
        wave = np.where(np.abs(wave) < cutoff, 0, wave)

    n = len(wave)
    step = np.round(np.arange(0, n+1-wl, step=int(wl - (ovlp * wl/100))))
    N = len(step)
    x = np.linspace(0, n/fs, num=N)
    y1 = stdft(wave=wave, fs=fs, wl=wl, zp=0, step=step, wn=wn)
    if bandpass:
        lowlimit = round((wl * bandpass[0]) / fs)
        uplimit = round((wl * bandpass[1]) / fs)
        y1[[x for x in range(y1.shape[0]) if x < lowlimit or x > uplimit], ] = 0
    maxi = np.max(y1, axis=0)
    y2 = np.argmax(y1, axis=0).astype('float') + 1
    y2[np.where(maxi == 0)[0]] = np.nan
    if clip:
        y2[np.where(maxi < clip)[0]] = np.nan
    y = (fs * (y2)) / (1000 * wl) - fs / (1000 * wl)
    return np.transpose([x, y])


def stdft(wave, fs, wl, step, zp=0, wn="hanning", scale=True, norm=False, complex=False):
    """Computes Short Term Discrete Fourier Transform based on R seewave.

    Args:
        wave (array): WAV array returned by scipy.io wavfile.read.
        fs (int): sampling frequency in HZ.
        wl (int): length of window.
        step (array): steps.
        zp (int, optional): zero-padding size. Defaults to 0.
        wn (str, optional): window name. Defaults to "hanning".
        scale (bool, optional): normalize to 1 the complete matrix. Defaults to True.
        norm (bool, optional): normalize to 1 each column (ie each FFT). Defaults to False.
        complex (bool, optional): no normalization. Defaults to False.

    Raises:
        ValueError: if zero-padding size is negative.

    Returns:
        array: short term fourier transforms.
    """
    if zp < 0:
        raise ValueError("zero-padding cannot be negative")
    if wn == "hanning":
        W = np.hanning(wl)
    elif wn == "blackman":
        W = np.blackman(wl)
    else:
        W = np.hamming(wl)

    z = np.zeros((wl, len(step)), dtype=np.complex128)
    for e, i in enumerate(step):
        z[:, e] = np.fft.fft(np.append(wave[i:(wl+i)]*W, np.zeros(zp)))

    z = z[0:((wl+zp) // 2)]  # to keep only the relevant frequencies
    z = z / (wl+zp)  # scaling by the original number of fft points

    if not complex:
        z = 2 * np.abs(z)  # multiplied by 2 to save the total energy
        if scale:
            if norm:  # normalise to 1 each column (ie each FFT)
                colmax = np.max(z, axis=0)
                z = (z.T / colmax).T
            else:
                z = z / np.max(z)  # normalise to 1 the complete matrix
    return z


def shannon_entropy(y):
    """Returns Shannon spectral entropy.

    Args:
        y (array): values.

    Returns:
        float: Shannon spectral entropy.
    """
    return -(y*np.log2(y)/np.log2(len(y))).sum()


def spectral_flatness(y):
    """Compute the spectral flatness (ratio between geometric and arithmetic means).

    Args:
        y (array): values.

    Returns:
        float: spectral flatness.
    """
    geometricMean = scipy.stats.mstats.gmean(abs(y))
    arithmeticMean = y.mean()
    return geometricMean / arithmeticMean


def spec(wave, fs=8000, wn="hanning", norm=True):
    """Returns frequency spectrum as per R seewave:spec.

    Args:
        wave (array): WAV array returned by scipy.io wavfile.read
        fs (int, optional): sampling frequency in HZ. Defaults to 8000.
        wn (str, optional): window name. Defaults to "hanning".
            Defaults to True.

    Returns:
        (numpy array): freq, amp
    """
    n = len(wave)
    if wn == "hamming":
        W = np.hamming(n)
    elif wn == "blackman":
        W = np.blackman(n)
    else:
        W = np.hanning(n)
    wave = wave * W

    y = np.abs(np.fft.rfft(wave))
    y = 2 * y[:(n//2)]
    if norm:
        y = y/np.max(y)
    x = (np.fft.rfftfreq(n, d=1/fs)/1000)[:n//2]
    return np.transpose([x, y])


def specprop(spec, fs=None, flim=None, khz=False):
    """Calculates a list of statistical properties of a frequency spectrum.
    Credit: https://github.com/primaryobjects/voice-gender/blob/master/sound.R

    Args:
        spec (array): data set obtained with spec.
        fs (int, optional): sampling frequency of spec (in Hz). Defaults to None.
        flim (tuple, optional): tuple of length 2 to specify the frequency limits
            of the analysis (in kHz) Defaults to None.
        khz (bool, optional): return Hz values in KHz. Defaults to False.

    Raises:
        ValueError: should be between 0 and half of sampling frequency

    Returns:
        dict: statistical properties of a frequency spectrum
    """
    if fs is None:
        fs = spec[len(spec) - 1, 0] * 2000 * len(spec)/(len(spec)-1)
    freq = spec[:, 0] * 1000
    spec = spec[:, 1]
    L = len(spec)
    wl = L * 2
    if flim:
        if flim[0] < 0 or flim[1] > fs/2:
            raise ValueError("'flim' should be between 0 and {0}/2".format(fs))
    else:
        flim = (0, (fs/2-fs/wl)/1000)
    g = (1000*wl/2) / (fs/2 - fs/wl)
    spec = spec[int((flim[0]*g)):int((flim[1]*g))]
    L = len(spec)

    # Amplitude
    amp = spec / spec.sum()
    amp_cumsum = np.cumsum(amp)
    z = amp - amp.mean()
    w = amp.std(ddof=1)

    # Frequency
    freq = np.linspace(flim[0] * 1000, flim[1] * 1000, num=L, retstep=False)

    # spectral stats I
    meanfreq = centroid = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - meanfreq) ** 2)))
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25

    # spectral stats II
    skew = ((z ** 3).sum() / (L - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (L - 1)) / w ** 4
    sh = shannon_entropy(amp)
    sfm = spectral_flatness(amp)
    # prec = fs/wl

    return {
        "meanfreq": meanfreq if not khz else meanfreq/1000,
        "sd": sd if not khz else sd/1000,
        "median": median if not khz else median/1000,
        "Q25": Q25 if not khz else Q25/1000,
        "Q75": Q75 if not khz else Q75/1000,
        "IQR": IQR if not khz else IQR/1000,
        "skew": skew,
        "kurt": kurt,
        "spent": sh,
        "sfm": sfm,
        "mode": mode if not khz else mode/1000,
        "centroid": centroid if not khz else centroid/1000,
    }


def specan(wave, fs=8000, wl=2048, threshold=5, duration=None, khz=True, ch=None):
    """Perform spectral analysis of WAV files extracting acoustic properties.

    Args:
        wave (string or array): WAV filename or numpy array.
        fs (int, optional): sampling frequency of spec (in Hz). Defaults to 8000.
        wl (int, optional): length of window. Defaults to 2048.
        threshold (int, optional): amplitude threshold for signal detection (in %). Defaults to 5.
        duration (int, optional): number of seconds from the beginning to consider. Defaults to None.
        khz (bool, optional): return Hz values in KHz. Defaults to True.
        ch (int, optional): channel number. Defaults to None.

    Returns:
        list: acoustic properties of channel(s).
    """
    if isinstance(wave, str):
        fs, wave = scipy.io.wavfile.read(wave)
    channels = []
    for chnum in range(2):
        if ch != chnum and ch is not None:
            channels.append({})
            continue
        if len(wave.shape) == 2:
            wave = wave[:, chnum]
        if duration:
            wave = wave[:fs * duration]
        songspec = spec(wave, fs)
        analysis = specprop(songspec, fs=fs, flim=(0, 280/1000), khz=khz)
        ff = fund(wave, fs, wl=wl, fmax=280, threshold=threshold, ovlp=50)[:, 1]
        meanfun = np.nanmean(ff)
        minfun = np.nanmin(ff)
        maxfun = np.nanmax(ff)
        y = dfreq(wave, fs, wl=wl, ovlp=0, bandpass=(0, 22000), threshold=threshold)[:, 1]
        meandom = np.nanmean(y)
        mindom = np.nanmin(y)
        maxdom = np.nanmax(y)
        dfrange = (maxdom - mindom)
        changes = np.abs(np.diff(y[~(np.isnan(y))]))
        modindx = 0 if mindom == maxdom else np.nanmean(changes) / dfrange
        analysis2 = {
            "meanfun": meanfun,
            "minfun": minfun,
            "maxfun": maxfun,
            "meandom": meandom,
            "mindom ": mindom,
            "maxdom": maxdom,
            "dfrange": dfrange,
            "modindx": modindx,
        }
        channels.append({**analysis, **analysis2})

    return channels


if __name__ == "__main__":
    import glob
    for file in glob.glob("data/*"):
        print(file)
        acoustics = specan(file, duration=10)
        for acoustic in acoustics:
            print("---")
            for k, v in acoustic.items():
                print("{0}: {1}".format(k, v))
