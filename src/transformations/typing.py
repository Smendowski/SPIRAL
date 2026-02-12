from typing_extensions import Annotated
from numpy.typing import NDArray
import numpy as np

Array1D = Annotated[NDArray[np.float32], "(TimeSteps: Height, Features: 1)"]
ArrayImage = Annotated[
    NDArray[np.float32], "(Channels: 3, TimeSteps: Height, Features: Width)"
]
