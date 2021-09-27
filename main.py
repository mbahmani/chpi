from tools.functions import clusering_per_algorithm
import os 
from tools.cvi import Validation
from tools.fANOVA_functions import *
from tools.helper import *

if __name__ == '__main__':
    
    # clusering_per_algorithm(path="datasets/", algorithm="Kmeans")
    # do_fanova(dataset_name="performance_data/Kmeans_results.csv",algorithm="kmeans",st=0,end=99)
   
    
    # for cls in {"kmeans"}:
    #     df=pd.read_csv("performance_data/"+cls+"_fANOVA_results.csv")
    #     total_ranks, marginal_contribution, _ = obtain_marginal_contributions(df)
    #     sorted_values, keys = determine_relevant(marginal_contribution, max_interactions=0)
    #     marginal_plots(sorted_values, keys, cls+" clustering_NMI_100")
    # for cls in {"dbscan"}:
    #     clusering_per_algorithm(path="datasets/", algorithm=cls)
        