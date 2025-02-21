"""Signal attributes testing parameters"""

from dasf_seismic_extras.attributes.signal import (
    RMS,
    RMS2,
    FirstDerivative,
    GradientMagnitude,
    HistogramEqualization,
    PhaseRotation,
    ReflectionIntensity,
    RescaleAmplitudeRange,
    SecondDerivative,
    TimeGain,
    TraceAGC,
)

attributes = [
    {
        "operator_cls": FirstDerivative,
        "v3_skip": True,
        "v3_skip": True,
    },
    {
        "operator_cls": SecondDerivative,
        "v3_skip": True,
    },
    {
        "operator_cls": HistogramEqualization,
        "v3_skip": True,
    },
    {
        "operator_cls": TimeGain,
        "v3_skip": True,
    },
    {
        "operator_cls": RescaleAmplitudeRange,
        "operator_params": {"min_val": 0.1, "max_val": 0.7},
        "v3_skip": True,
    },
    {
        "operator_cls": RMS,
        "v3_skip": True,
    },
    {
        "operator_cls": RMS2,
        "v3_skip": True,
    },
    {
        "operator_cls": TraceAGC,
        "v3_skip": True,
    },
    {
        "operator_cls": GradientMagnitude,
        "v3_skip": True,
    },
    {
        "operator_cls": ReflectionIntensity,
        "v3_skip": True,
    },
    {
        "operator_cls": PhaseRotation, 
        "operator_params": {"rotation": 90},
        "v3_skip": True,
    },
]
