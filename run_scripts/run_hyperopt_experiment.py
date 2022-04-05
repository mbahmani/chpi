from tools import hpi_verification
from sklearn.cluster import KMeans, AffinityPropagation, \
    AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, OPTICS
import pandas as pd
import json
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    path = "./datasets/"
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    all_datasets = len(all_files)
    hp_trials_datasets = {}
    ranked_hpi_datasets = {}
    avg_losses_dataset= {}
    for index, file in enumerate(all_files):
        print('Dataset {}({}) out of {} \n'.format(index + 1, file, all_datasets), flush=True)

        file_path = path+file

        df = pd.read_csv(file_path, index_col=None, header=0)

    # excluding the response variable
        X = df.iloc[:, :-1]

        # selecting the response variable
        y = df.iloc[:, -1]

        hpiverfication = hpi_verification.HPIVerification(
            n_iter_opt=40,
            X=X,
            y=y,
            algorithm="kmeans",
            )

        hp_trials_dict = hpiverfication.hyper_opt_run()
        hp_trials_datasets[file] = hp_trials_dict

        avg_losses_dataset[file] = hpiverfication.hpi_avg_lossess_dataset(hp_trials_dict)

    avg_losses = hpiverfication.hpi_avg_lossess(avg_losses_dataset, all_files)

    ranked_hpis, hpi_names = hpiverfication.rank_hpi(avg_losses)
    
  
    hpiverfication.plot(ranked_hpis, hpi_names)
    print("done!")
