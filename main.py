from tools.functions import clusering_per_algorithm
import os 
from tools.cvi import Validation

if __name__ == '__main__':
    
    clusering_per_algorithm(path="datasets/", algorithm="Kmeans")