### Signal Process Attributes

|       **Atribute**         | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:--------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|     First Derivative       |    Ready   |    X    |    X    |       X       |       X       |
|    Second Derivative       |    Ready   |    X    |    X    |       X       |       X       |
|   Histogram Equalization   |  Unstable  |    X    |    X    |       X       |       X       |
|        Time Gain           |  Unstable  |    X    |    X    |       X       |       X       |
|   Rescale Amplitude Range  |    Ready   |    X    |    X    |       X       |       X       |
|           RMS^1            |    Ready   |    X    |    X    |       X       |       X       |
|        Trace AGC           |    Ready   |    X    |    X    |       X       |       X       |
|    Gradient Magnitude      |    Ready   |    X    |    X    |       X       |       X       |
|   Reflection Intensity     |  Unstable  |    X    |         |       X       |               |
|     Phase Rotation         |    Ready   |    X    |    X    |       X       |       X       |

#### Observations:

* The attribute *Reflection Intensity* requires `trapz()` function. It is only available in CuPy release 11.0.0.

^1 RMS has two implementations: One using CuPy and another one using CUDA code.
