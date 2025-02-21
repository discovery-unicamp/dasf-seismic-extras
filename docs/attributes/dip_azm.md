### Dip and Azimuth Attributes

From: https://doi.org/10.1190/geo2018-0530.1

Seismic volumetric dip and azimuth are widely used in assisting seismic interpretation to depict geologic structures such as chaotic slumps, fans, faults, and unconformities. Current popular dip and azimuth estimation methods include the semblance-based multiple window scanning (MWS) method and gradient structure tensor (GST) analysis. However, the dip estimation accuracy using the semblance scanning method is affected by the dip of seismic reflectors. The dip estimation accuracy using the GST analysis is affected by the analysis window centered at the analysis point. We have developed a new algorithm to overcome the disadvantages of dip estimation using MWS and GST analysis by combining and improving the two methods. The algorithm first obtains an estimated "rough" dip and azimuth for reflectors using the semblance scanning method. Then, the algorithm defines a window that is "roughly" parallel to the local reflectors using the estimated rough dip and azimuth. The algorithm next estimates the dip and azimuth of the reflectors within the analysis window using GST analysis. To improve the robustness of GST analysis to noise, we used analytic seismic traces to compute the GST matrix. The algorithm finally uses the Kuwahara window strategy to determine the dip and azimuth of local reflectors. To illustrate the superiority of this algorithm, we applied it to the F3 block poststack seismic data acquired in the North Sea, Netherlands. The comparison indicates that the seismic volumetric dips estimated using our method more accurately follow the local seismic reflectors than the dips computed using GST analysis and the semblance-based MWS method.

|       **Atribute**         | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:--------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|      Gradient Dips         |    Ready   |    X    |    X    |       X       |       X       |
| Gradient Structure Tensor  |    Ready   |    X    |    X    |       X       |       X       |
|       GST 2D Dip           |    Ready   |    X    |    X    |       X       |       X       |
|       GST 3D Dip           |    Ready   |    X    |    X    |       X       |       X       |
|       GST 3D Azm           |    Ready   |    X    |    X    |       X       |       X       |

