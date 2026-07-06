import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        list1=[]
        for i in z:
            val=1/(1+np.exp(0-i))
            list1.append(round(val,5))
        # return np.round(your_answer, 5)
        return list1

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        list1=[]
        for i in z:
            list1.append(max(0.0,i))
        # Formula: max(0, z) element-wise
        return list1
