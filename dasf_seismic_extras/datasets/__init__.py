from dasf_seismic_extras.datasets.base import *  # noqa
from dasf_seismic_extras.datasets.blake import *  # noqa
from dasf_seismic_extras.datasets.delft import *  # noqa
from dasf_seismic_extras.datasets.f3 import *  # noqa
from dasf_seismic_extras.datasets.kahu import *  # noqa
from dasf_seismic_extras.datasets.kerry import *  # noqa
from dasf_seismic_extras.datasets.opunake import *  # noqa
from dasf_seismic_extras.datasets.panoma_council_grove_field_well_logs import *  # noqa
from dasf_seismic_extras.datasets.parihaka import *  # noqa
from dasf_seismic_extras.datasets.poseidon import *  # noqa
from dasf_seismic_extras.datasets.stratton import *  # noqa
from dasf_seismic_extras.datasets.teapot import *  # noqa
from dasf_seismic_extras.datasets.waihapa import *  # noqa
from dasf_seismic_extras.datasets.waka import *  # noqa

files = [
           # Base Dataset imports
           "DatasetSEGY",
           "DatasetSEISNC",
           # Blake Ridge Hydrates dataset
           "BlakeRidgeHydrates",
           # Delft dataset
           "Delft",
           # F3 Netherlands dataset
           "F3",
           # Kahu dataset
           "Kahu",
           # Kerry dataset
           "Kerry",
           # Opunake dataset
           "Opunake",
           # Panoma well logs dataset
           "PanomaCouncilGroveFieldFaciesWellLogs",
           # Parihaka dataset
           "ParihakaFull",
           "ParihakaTrain",
           "ParihakaTrainLabels",
           "ParihakaDataTest1",
           "ParihakaDataTest2",
           # Poseidon dataset
           "Poseidon",
           # Stratton dataset
           "Stratton3D",
           # Teapot dataset
           "Teapot3D",
           "Teapot3DCMP",
           "Teapot3DRawCMP",
           # Waihapa dataset
           "Waihapa",
           # Waka dataset
           "Waka",
]

__all__ = files
