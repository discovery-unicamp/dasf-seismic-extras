"""Dip Azm attributes testing parameters"""

from dasf_seismic_extras.attributes.dip_azm import (
    GradientDips,
    GradientStructureTensor,
    GradientStructureTensor2DDips,
    GradientStructureTensor3DAzm,
    GradientStructureTensor3DDip,
)

attributes = [
    {
        "operator_cls": GradientDips,
        "v1_outputs": 2,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": GradientStructureTensor,
        "operator_params": {
            "kernel": (2, 2, 2),
        },
        "v1_outputs": 6,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": GradientStructureTensor2DDips,
        "v1_outputs": 2,
        "v2_precision": 4,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": GradientStructureTensor3DDip,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": GradientStructureTensor3DAzm,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
]
