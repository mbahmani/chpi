"""
    Hyperparameter importance verifcation with hyperopt.
    In this file, user can find all the materails,
    in order to verfiy the result of fANOVA.
    by fixing each hyperparameter once to
    see what would be the rank of each one.
"""
from hyperopt import hp, tpe, fmin, Trials, rand
import numpy as np
import sys
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, \
    DBSCAN, MeanShift, SpectralClustering, OPTICS
from scipy.stats import rankdata


class HPIVerification:
    """Base class for hyperparameter importance verfication for clustering."""

    def __init__(self, n_iter_opt, X, y, algorithm):
        self.n_iter_opt = n_iter_opt
        self.X = X
        self.y = y
        self.algorithm = algorithm

    N = 2
    # define the ML models
    models = {
        'kmeans': KMeans(),
        "agglomerativeclustering": AgglomerativeClustering(),
        "dbscan": DBSCAN(),
        "optics": OPTICS(),
        "meanshift": MeanShift(),
        "spectralclustering": SpectralClustering(),
         }

    # create hyperparameter search space for each hyperparameter.
    params_kmeans = {
            "n_clusters": np.random.choice(np.arange(20, 23, 1), N),
            "init": np.random.choice(['k-means++', 'random'], N),
            "max_iter": np.random.choice(np.arange(50, 501, 1), N),
            "algorithm": np.random.choice(['auto', 'full', 'elkan'], N),
            "n_init": np.random.choice(np.arange(2, 30, 1), N),
            'tol': np.random.uniform(10 ** (-5), 10 ** (-1), N),
                    }

    params_agglomerativeclustering = {
                                    "n_clusters": np.random.choice(np.arange(2, 40, 1), N),
                                    "affinity": np.random.choice(['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], N),
                                    "linkage": np.random.choice(['ward', 'complete', 'average', 'single'], N),
                                    "compute_full_tree": np.random.choice(['auto', 'True', 'False'], N),
                                }

    params_dbscan = {
                        'eps': np.random.uniform(0.01, 1, N),
                        "min_samples": np.random.choice(np.arange(2, 20, 1), N),
                        "metric": np.random.choice(['euclidean', 'l1', 'l2', 'manhattan'], N),
                        "algorithm": np.random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'], N),
                        "leaf_size": np.random.choice(np.arange(2, 50, 1), N),
                    }

    params_optics = {
                    'max_eps': np.random.uniform(0.01, sys.maxsize, N),
                    'eps': np.random.uniform(0.01, sys.maxsize, N),
                    "min_samples": np.random.choice(np.arange(2, 20, 1), N),
                    "cluster_method": np.random.choice(['xi', 'dbscan'], N),
                    'xi': np.random.uniform(0.001, 1, N),
                    "metric": np.random.choice(['euclidean', 'l1', 'l2', 'manhattan'], N),
                    "algorithm": np.random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'], N),
                    "leaf_size": np.random.choice(np.arange(2, 50, 1), N),
                    }

    params_meanshift = {
                        'bin_seeding': np.random.choice(['True', 'False'], N),
                        "cluster_all": np.random.choice(['True', 'False'], N),
                        "max_iter": np.random.choice(np.arange(50, 501, 1), N),
                        'bandwidth': np.random.uniform(0.1, 2.5, N),

                    }

    params_spectralclustering = {
                                "n_clusters": np.random.choice(np.arange(2, 40, 1), N),
                                "n_init":  np.random.choice(np.arange(2, 30, 1), N),
                                "affinity":  np.random.choice(['nearest_neighbors', 'rbf'], N),
                                "n_neighbors": np.random.choice(np.arange(2, 40, 1), N),
                                "assign_labels": np.random.choice(['kmeans', 'discretize'], N),
                            }

    parameters = {
                    'kmeans': params_kmeans,
                    "agglomerativeclustering": params_agglomerativeclustering,
                    "dbscan": params_dbscan,
                    "optics": params_optics,
                    "meanshift": params_meanshift,
                    "spectralclustering": params_spectralclustering,
                }

    hyperopt_params_kmeans = {
                    "n_clusters": hp.choice("n_clusters", np.arange(2, 40, 1)),
                    "init": hp.choice("init", ['k-means++', 'random']),
                    "max_iter": hp.choice("max_iter", np.arange(50, 501, 1)),
                    "algorithm": hp.choice("algorithm", ['auto', 'full', 'elkan']),
                    "n_init": hp.choice("n_init", np.arange(2, 30, 1)),
                    'tol': hp.uniform("tol", 10 ** (-5), 10 ** (-1)),
                }

    hyperopt_parameters = {
                    'kmeans': hyperopt_params_kmeans,
                    "agglomerativeclustering": params_agglomerativeclustering,
                    "dbscan": params_dbscan,
                    "optics": params_optics,
                    "meanshift": params_meanshift,
                    "spectralclustering": params_spectralclustering,
                }

    def train_evaluate(self, params):
        model = self.models[self.algorithm]
        model.set_params(**params)
        predictions_tr = model.fit_predict(self.X)
        score = normalized_mutual_info_score(self.y, predictions_tr)
        return score

    def objective(self, params):
        return -1.0 * self.train_evaluate(params)

    def hp_search(self, hpi):
        """Filter the search space based on hpi.

        Args:
            hpi (str): the hyperparmeter which we want to fix it during optimization.

        Returns:
            k_run(ndarray): The generated random samples.

        """
        k_run = self.parameters[self.algorithm][hpi]
        if hpi in "init":
            k_run = np.random.choice(['k-means++', 'random'], self.N)
            k_run_str = [str(n) for n in k_run]
            return k_run_str

        return k_run

    def hyper_opt_run(self):
        """Returns the resluts of applying hyperopt by fixing
        at the time one hyperparameter and tuning across other hyperparameters.

        Returns:
            hp_trials_dict(dict): A dictionary which
            keeps all the result for K diffrent configuration
            for fixed hyperparamter and n different itereation 
            for optimization.
        """
        hp_trials_dict = {}

        hpi_dict = self.parameters[self.algorithm]
        for hpi in hpi_dict.keys():
            SEARCH_PARAMS = self.hyperopt_parameters[self.algorithm]
            SEARCH_PARAMS_dict = SEARCH_PARAMS.copy()
            SEARCH_PARAMS_dict.pop(hpi)
            k_run = self.hp_search(hpi)

            trials_lst = []
            best_lst = []

            for i in range(0, self.N):

                FIXED_PARAMS = {
                    hpi: k_run[i],
                }

                params = {**SEARCH_PARAMS_dict, **FIXED_PARAMS}
                trials = Trials()
                best = fmin(self.objective,
                            space=params,
                            algo=tpe.suggest,
                            max_evals=self.n_iter_opt,
                            trials=trials)

                trials_lst.append(trials.results)
                best_lst.append(best)

            hp_trials_dict[hpi] = trials_lst

        return hp_trials_dict

    def hpi_avg_lossess_dataset(self, hp_trials_lst):
        """
        calculates average lossess based on K run and returns ranked hpi.

        Args:
            hp_trials_lst (dict): A dictionary which 
            keeps all the result for K diffrent configuration for fixed hyperparamter and
            n different itereation for optimization.

        Returns:
            avg_losses(dict): return average lossess based on fixed each hyperparameter once per time.
        """
        losses = {}
        avg_losses = {}
        lst_hpi_losses = {}

        hpi_dict = self.parameters[self.algorithm]

        for hpi in hpi_dict.keys():
            lst_hpi_losses = hp_trials_lst[hpi]
            losses[hpi] = [[lst_hpi_losses[y][x]["loss"] for x in range(self.n_iter_opt)] for y in range(self.N)]
            avg_losses[hpi] = [sum(x)/len(losses[hpi]) for x in zip(*losses[hpi])]

        return avg_losses

    def hpi_avg_lossess(self, dicts, db_name):
        len_db = len(db_name)
        merged = dicts[db_name[0]]
        dicts.pop(db_name[0])
        db_name.pop(0)
        for d in db_name:
            for hp in dicts[d]:
                merged[hp] = [sum(k)  for k in zip(dicts[d][hp], merged[hp])]
        
        for hp in merged:
            merged[hp] = [item / len_db for item in merged[hp]]
            
        return merged
    # def hpi_avg_lossess(self, avg_losses_dataset):
    #     for dataset in avg_losses_dataset:
    #         print(avg_losses_dataset[dataset])
    
    def rank_hpi(self, avg_losses):
        """_summary_

        Args:
            avg_losses (dict): _description_

        Returns:
            ranked_hpi: _description_
        """
        ranked_hpis = []
        ranked_hpis = rankdata([avg_losses[key] for key in avg_losses.keys()], method="ordinal", axis=0)
        hpi_names = list(avg_losses.keys())
        return ranked_hpis, hpi_names

    def plot(self, ranked_hpis, hpi_names):
        """draw hyperparameters based on thier rank.

        Args:
            ranked_hpis (list): It contains rank of each hyperparameters.
            hpi_names (list): name of hyperparameters.
        Returns:
            show the plot
        """
        for ranked_hpi, hpi_name in zip(ranked_hpis, hpi_names):
            plt.plot(range(0, self.n_iter_opt), ranked_hpi, label=str(hpi_name))

        plt.legend()
        plt.show()
        plt.savefig('plotttttttt.png')
        return print("plot was drawn! ")
