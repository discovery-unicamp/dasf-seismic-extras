### Edge Detection Attributes

References:

- "New Edge Detection Methods for Seismic Interpretation" - Brian Russell and Claude Ribordy.
 

|       **Atribute**        | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:-------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|        Semblance^1        |    Ready   |    X    |    X    |       X       |       X       |
|       EIG Complex         |    Ready   |    X    |         |       X       |               |
| Gradient Structure Tensor |    Ready   |    X    |    X    |       X       |       X       |
|          Chaos            |    Ready   |    X    |    X    |       X       |       X       |
|     Volume Curvature      |    Ready   |    X    |    X    |       X       |       X       |

#### Observations:

* The attribute *EIG Complex* requires function `eigvals()` which is not implemented in CuPy yet.

^1 Semblance has two implementations: One using CuPy and another one using CUDA code.
