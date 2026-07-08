import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        predictions=np.dot(X,weights)
        return np.round(predictions,5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        total=0
        for i in range(len(model_prediction)):
            total=total+((model_prediction[i]-ground_truth[i])**2)
        mse=total/len(model_prediction)
        return np.round(mse[0],5)
        # Round to 5 decimal places
        
