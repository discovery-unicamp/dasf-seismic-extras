### Frequency Attributes

|          **Atribute**            | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:--------------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|          Lowpass Filter          |  Unstable  |    X    |         |       X       |               |
|         Highpass Filter          |  Unstable  |    X    |         |       X       |               |
|         Bandpass Filter          |  Unstable  |    X    |         |       X       |               |
|           CWT Ricker             |    Ready   |    X    |    X    |       X       |       X       |
|           CWT Ormsby             |    Ready   |    X    |    X    |       X       |       X       |
|   Spectral Decomposition STFT    |    Ready   |    X    |    X    |       X       |       X       |
| Spectral Decomposition Stockwell |    Ready   |    X    |    X    |       X       |       X       |

#### Observations:

* The attributes *Lowpass Filter*, *Highpass Filter* and *Bandpass Filter* requires `iirfilter()` which is not developed inside `cusignal` yet.
