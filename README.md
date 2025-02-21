# DASF module for Seismic (extras)

[![Continuous Test](https://github.com/discovery-unicamp/dasf-seismic-extras/actions/workflows/ci.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/ci.yaml)
[![Commit Check Policy](https://github.com/discovery-unicamp/dasf-seismic-extras/actions/workflows/commit-check.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/commit-check.yaml)

DASF Seismic is a module dedicated to seismic operations. The project contains 
datasets that are capable to manipulate formats like SEG-Y and 
[Seisnc](https://segysak.readthedocs.io/en/latest/seisnc-standard.html), for 
example. It is also possible to calculate the most common seismic atrtibutes 
and many other common transformations.

Part of the attributes implementation was based on [d2geo](https://github.com/yohanesnuwara/d2geo) 
project originally, but this one has other important and recent attributes and 
the GPU support.

This project is based on top of [dasf-seismic](https://github.com/lmcad-unicamp/dasf-seismic).

## Install

The installation can be done using `poetry install` or `pip install` with 
wheel installed and it requires [**dasf-core**](https://github.com/discovery-unicamp/dasf-core)
installed first.

```bash
pip3 install .
```

# Attributes

For further revision of what attribute is implemented see the Documentation. 
The list of implemented attributes is following:

- [Complex Trace Attributes](docs/attributes/complex_trace.md)
- [Signal Process Attributes](docs/attributes/signal_process.md)
- [Frequency Attributes](docs/attributes/frequency.md)
- [Edge Detection Attributes](docs/attributes/edge_detection.md)
- [Dip and Azimuth Attributes](docs/attributes/dip_azm.md)
- [Noise Reduction in 3D Seismic Data](docs/attributes/noise_reduction.md)
