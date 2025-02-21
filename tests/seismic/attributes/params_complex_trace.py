"""Complex Trace attributes testing parameters"""

from dasf_seismic_extras.attributes.complex_trace import (
    AmplitudeAcceleration,
    ApparentPolarity,
    CosineInstantaneousPhase,
    DominantFrequency,
    Envelope,
    FrequencyChange,
    Hilbert,
    InstantaneousBandwidth,
    InstantaneousFrequency,
    InstantaneousPhase,
    QualityFactor,
    RelativeAmplitudeChange,
    ResponseAmplitude,
    ResponseFrequency,
    ResponsePhase,
    Sweetness,
)

attributes = [
    {
        "operator_cls": Hilbert,
        "dtypes": {
            "float32": "complex64",
            "float64": "complex128",
        },
        "v3_skip": True,
    },
    {
        "operator_cls": Envelope,
        "v3_skip": True,
    },
    {
        "operator_cls": InstantaneousPhase,
        "v3_skip": True,
    },
    {
        "operator_cls": CosineInstantaneousPhase,
        "v3_skip": True,
    },
    {
        "operator_cls": RelativeAmplitudeChange,
        "v3_skip": True,
    },
    {
        "operator_cls": AmplitudeAcceleration,
        "v3_skip": True,
    },
    {
        "operator_cls": InstantaneousFrequency,
        "v3_skip": True,
    },
    {
        "operator_cls": InstantaneousBandwidth,
        "v3_skip": True,
    },
    {
        "operator_cls": DominantFrequency,
        "v3_skip": True,
    },
    {
        "operator_cls": FrequencyChange,
        "v3_skip": True,
    },
    {
        "operator_cls": Sweetness,
        "v3_skip": True,
    },
    {
        "operator_cls": QualityFactor,
        "v3_skip": True,
    },
    {
        "operator_cls": ResponsePhase,
        "v3_skip": True,
    },
    {
        "operator_cls": ResponseFrequency,
        "v3_skip": True,
    },
    {
        "operator_cls": ResponseAmplitude,
        "v3_skip": True,
    },
    {
        "operator_cls": ApparentPolarity,
        "v3_skip": True,
    },
]
