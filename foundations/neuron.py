import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        z=np.dot(x,w)+b
        # activation: "sigmoid" or "relu"
        if activation=="sigmoid":
            res=1/(1+np.exp(-z))
        elif activation=="relu":
            res=max(0.0,z)
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        return round(res,5)
