"""Edge Detection attributes testing parameters"""

from dasf_seismic_extras.attributes.dip_azm import GradientDips

from dasf_seismic_extras.attributes.edge_detection import (
    Chaos,
    Coherence,
    EigComplex,
    Semblance,
    Semblance2,
    StructuredSemblance,
    VolumeCurvature,
)

attributes = [
    {
        "operator_cls": Semblance,
        "v2_slice": (15, 15, 15),
        "v2_in_shape_chunks": (5, 5, 5),
        "v3_skip": True,
    },
    {
        "operator_cls": Semblance2,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": EigComplex,
        "v3_skip": True,
    },
    {
        "operator_cls": Chaos,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": Coherence,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": StructuredSemblance,
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
    {
        "operator_cls": VolumeCurvature,
        "v1_inputs": 2,
        "v1_outputs": 6,
        "v2_preprocessing": [(GradientDips, (0, 1))],
        "v2_in_shape_chunks": (20, 20, 20),
        "v3_skip": True,
    },
]
