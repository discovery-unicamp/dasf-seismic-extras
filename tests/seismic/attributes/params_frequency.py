"""Frequency attributes testing parameters"""

from dasf_seismic_extras.attributes.frequency import (
    BandpassFilter,
    CWTOrmsby,
    CWTRicker,
    HighpassFilter,
    LowpassFilter,
)

attributes = [
    {
        "operator_cls": LowpassFilter,
        "operator_params": {"freq": 30},
        "v1_in_shape": (100, 100, 100),
        "v1_in_shape_chunks": (50, 50, 50),
        "v3_skip": True,
    },
    {
        "operator_cls": HighpassFilter,
        "operator_params": {"freq": 70},
        "v1_in_shape": (100, 100, 100),
        "v1_in_shape_chunks": (50, 50, 50),
        "v3_skip": True,
    },
    {
        "operator_cls": BandpassFilter,
        "operator_params": {"freq_lp": 0.3, "freq_hp": 0.7},
        "v1_in_shape": (100, 100, 100),
        "v1_in_shape_chunks": (50, 50, 50),
        "v3_skip": True,
    },
    {
        "operator_cls": CWTRicker,
        "operator_params": {"freq": 70},
        "v1_in_shape": (100, 100, 100),
        "v1_in_shape_chunks": (50, 50, 50),
        "v3_skip": True,
    },
    {
        "operator_cls": CWTOrmsby,
        "operator_params": {"freqs": (20, 40, 60, 80)},
        "v1_in_shape": (100, 100, 100),
        "v1_in_shape_chunks": (50, 50, 50),
        "v3_skip": True,
    },
]
