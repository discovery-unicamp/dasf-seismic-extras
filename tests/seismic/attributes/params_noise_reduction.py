"""Noise reduction attributes testing parameters"""

from dasf_seismic_extras.attributes.noise_reduction import Convolution, Gaussian, Median

attributes = [
    {
        "operator_cls": Gaussian,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": Median,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": Convolution,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
]
