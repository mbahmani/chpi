import pandas as pd 
import numpy as np
import os
import glob
import warnings
import time
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  adjusted_rand_score, silhouette_score, rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AffinityPropagation,AgglomerativeClustering, DBSCAN,MeanShift,SpectralClustering
from sys import stdout


N= 100

params_kmeans={
"n_clusters": np.random.choice(np.arange(2, 40, 1), N),
"init": np.random.choice(['k-means++','random'], N),
"max_iter": np.random.choice(np.arange(50, 501, 1), N),
"algorithm": np.random.choice(['auto' ,'full', 'elkan'], N), 
"n_init": np.random.choice(np.arange(2, 30, 1), N),
'tol': np.random.uniform(10 ** (-5), 10 ** (-1), N),
}

params_affinitypropagation={
"affinity": np.random.choice(['euclidean','precomputed'], N),
"max_iter": np.random.choice(np.arange(50, 501, 1), N),
"convergence_iter": np.random.choice(np.arange(2, 30, 1), N),
'damping': np.random.uniform(0.5, 0.999, N),
}

# we did not add distance_threshold as hyperparameter
params_agglomerativeclustering={
"n_clusters": np.random.choice(np.arange(2, 40, 1), N),
"affinity": np.random.choice(['euclidean','l1','l2','manhattan','cosine'], N),
"linkage": np.random.choice(['ward', 'complete', 'average', 'single'], N),
"compute_full_tree": np.random.choice(['auto', 'True', 'False'], N),
}

params_dbscan={
'eps': np.random.uniform(0.01, 1, N),    
"min_samples": np.random.choice(np.arange(2, 20, 1), N),
"metric": np.random.choice(['euclidean','l1','l2','manhattan'], N),
"algorithm": np.random.choice(['auto','ball_tree','kd_tree','brute'], N),
"leaf_size": np.random.choice(np.arange(2, 50, 1), N),
}

params_meanshift={
'bin_seeding': np.random.choice(['True','False'], N), 
"cluster_all": np.random.choice(['True','False'], N), 
"max_iter": np.random.choice(np.arange(50, 501, 1), N),
'bandwidth': np.random.uniform(0.1, 2.5,N),
 
}

params_spectralclustering={
"n_clusters": np.random.choice(np.arange(2, 40, 1), N),
# "eigen_solver":  np.random.choice(['arpack','lobpcg','amg'], N),
# "n_components": np.random.choice(np.arange(2, 40, 1), N),
"n_init":  np.random.choice(np.arange(2, 30, 1), N),
"affinity":  np.random.choice(['nearest_neighbors','rbf'], N),
"n_neighbors": np.random.choice(np.arange(2, 40, 1), N),
"assign_labels":np.random.choice(['kmeans','discretize'], N),
}


parameters ={
    'Kmeans' : params_kmeans,
    "affinitypropagation" : params_affinitypropagation,
    "agglomerativeclustering" : params_agglomerativeclustering,
    "dbscan" : params_dbscan,
    "meanshift" : params_meanshift,
    "spectralclustering" : params_spectralclustering,

      
}

models = {
    'Kmeans': KMeans(),
    "affinitypropagation" : AffinityPropagation(),
    "agglomerativeclustering" : AgglomerativeClustering(),
    "dbscan" : DBSCAN(),
    "meanshift" : MeanShift(),
    "spectralclustering" : SpectralClustering(),
}







def clusering_per_algorithm(path, algorithm):
    """ Fit diffrent model across many datasets

    Keyword Arguments:
        path {str} -- [description] (default: {""})
        algorithm {str} -- [description] (default: {"kmeans"})
    """
    warnings.filterwarnings("ignore")
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    all_datasets = len(all_files)
    

    results = pd.DataFrame()
    start_all = time.perf_counter()

    for index, file in enumerate(all_files):
        print('Dataset {}({}) out of {} \n'.format(index + 1, file, all_datasets), flush=True)
        try:
            file_logs = clustering_per_dataset(path,file, algorithm, models, parameters)
            results = pd.concat([results, file_logs], axis=0)

            results.to_csv('performance_data/{}_results.csv'.format(algorithm),
                            header=True,
                            index=False)
        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))
    end_all = time.perf_counter()
    time_taken = (end_all - start_all) / 3600
    stdout.write("Performance data is collected! \n ")
    print('Total time: {} hours'.format(time_taken))


def clustering_per_dataset(path, file, algorithm, models,parameters):
    """
    Obtaining performance information for each random configuration on the given dataset for the specific clustering algorithm.

    Arguments:
        path {[str]} -- [path of the dataset]
        file {[str]} -- [dataset name]
        algorithm {[type]} -- [description]
        models {[type]} -- [description]
        parameters {[type]} -- [description]
    """
    path=path + file
    data = pd.read_csv(path,
                index_col=None,
                header=0,
                na_values='?')
    
    # making the column names lower case
    data.columns = map(str.lower, data.columns)          

    # removing an id column if exists
    if 'id' in data.columns:
        data = data.drop('id', 1)

    # remove columns with only NaN values
    empty_cols = ~data.isna().all()
    data = data.loc[:, empty_cols]

    # identifying numerical and categorical features
    cols = set(data.columns)
    num_cols = set(data._get_numeric_data().columns)
    categorical_cols = list(cols.difference(num_cols))

    # data imputation for categorical features
    categ_data = data[categorical_cols]
    data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])

    # defining the random configurations
    combinations = get_combinations(parameters, algorithm)

    # data imputation for numeric features
    if data.isna().values.any():

        imputation_types = ['mean', 'median', 'mode']

        final_logs = pd.DataFrame()
        
        imputed_data = data.copy()

        for index, num_imput_type in enumerate(imputation_types):
            print('{}'.format(num_imput_type))

            imputed_data[list(num_cols)] = numeric_impute(data, num_cols, num_imput_type)

            # logs per imputation method
            logs = get_logs(imputed_data, num_imput_type, algorithm,
                            file, combinations, models)

            final_logs = pd.concat([final_logs, logs], axis=0)

    else:
        num_imput_type = "None"

        final_logs = get_logs(data, num_imput_type, algorithm,
                            file, combinations, models)

    return final_logs


def get_logs(data, num_imput_type, algorithm, file, combinations, models):
    """
    Gathers the performance data for each random configuration 
    on the given dataset for the specified ML algorithm
    
    Inputs:
            data - (DataFrame) dataset, where the last column 
                               contains the response variable 
            num_imput_type - (str or None) imputation type that takes 
                              one of the following values
                              {'mean', 'median', 'mode', None}
            
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
            file - (str) name of the dataset
            
            combinations - (DataFrame) contains the random configurations
                            of the given algorithm
            
            models - (dict) key: algorithm, 
                            value: the class of the algorithms
            
    Outputs: 
            logs - (DataFrame) performance data
            
    """
    # excluding the response variable
    X = data.iloc[:, :-1]

    # selecting the response variable
    y = data.iloc[:, -1]

    # one-hot encoding
    X = pd.get_dummies(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    num_labels = len(np.unique(y))



    # scaling the input 
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaled = True
   
    
    logs = combinations.copy()
    n_comb = logs.shape[0]
    logs.insert(loc=0, column='dataset', value=file)
    logs['imputation'] = num_imput_type
    
    ## check being real or synthetic for dataset
    # remove .csv
    file_name = file[:-4]
    real_dataset_check = file_name[-3:]
    if real_dataset_check=="_RL":
        logs['dataset_type'] = "Real"
    else:
        logs['dataset_type'] = "Synthetic"
    
    for index in range(n_comb):
        print('{}/{}'.format(index + 1, n_comb))
        
        params = dict(zip(combinations.columns,
                      list(combinations.iloc[index,:])))
        
        model = models[algorithm]
        model.set_params(**params)

       


        # fit and prediction 
        try:
            start_tr = time.perf_counter()
            predictions_tr = model.fit_predict(X)
            end_tr = time.perf_counter()
            train_time = end_tr - start_tr
        except:
            start_tr = 0.00
            predictions_tr = 0.00
            end_tr = 0.00
            train_time = end_tr - start_tr
        

         
        try:
            rnd_score = rand_score(y, predictions_tr)
        except: 
            rnd_score=0
        
        try:
            adj_rnd_score=adjusted_rand_score(y, predictions_tr)
        except: 
            adj_rnd_score=0
        
        try:
            sil_score=silhouette_score(X, predictions_tr)
        except: 
            sil_score=0
        
        try:
            adj_mut_score=adjusted_mutual_info_score(y, predictions_tr)
        except: 
            adj_mut_score=0
        
        try:
            nrm_mut_score=normalized_mutual_info_score(y, predictions_tr)
        except: 
            nrm_mut_score=0
        
        
        stdout.flush()

        logs.loc[index, "Train_time"] = np.mean(train_time)
        logs.loc[index, 'rand_score'] = np.mean(rnd_score)
        logs.loc[index, 'adjusted_rand_score'] = np.mean(adj_rnd_score)
        logs.loc[index, 'silhouette_score'] = np.mean(sil_score)
        logs.loc[index, 'adjusted_mutual_info_score'] = np.mean(adj_mut_score)
        logs.loc[index, 'normalized_mutual_info_score'] = np.mean(nrm_mut_score)
        

        print('\n')

    return logs


def get_combinations(parameters, algorithm):
    """
    Creates a DataFrame of the random configurations
    of the given algorithm
    
    Inputs:
            parameters - (dict) key: algorithm
                                value: the configuration space of 
                                       the algorithm
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
            
    Outputs: 
            combinations - (DataFrame) realizations of the 
                            random configurations
            
    """
    param_grid = parameters[algorithm]
    combinations = pd.DataFrame(param_grid)
    return combinations


def other_metrics(y_true, predictions, multilabel):
    """
    Treating the case of multiple labels for 
    computing performance measures suitable for 
    binary labels
    
    Inputs:
            y_true - (array or sparse matrix) true values of the 
                      labels
            predictions - (array or sparse matrix) predicted values of the 
                      labels 
            multilabel - (boolean) specifies if there are 
                          multiple labels (True)
    Outputs: 
            f1, recall, precision, auc - (float) performace metrics
            
    """
    
    if multilabel:
        f1 = f1_score(y_true, predictions, average='micro')
        recall = recall_score(y_true, predictions, average='micro')
        precision = precision_score(y_true, predictions, average='micro')
        auc = roc_auc_score(y_true, predictions, average='micro')
    else:
        f1 = f1_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        auc = roc_auc_score(y_true, predictions)

    return f1, recall, precision, auc


def numeric_impute(data, num_cols, method):
    """
    Performs numerical data imputaion based 
    on the given method
    
    Inputs:
            data - (DataFrame) dataset with missing 
                     numeric values 
            num_cols - (set) numeric column names
            method - (str) imputation type that takes 
                              one of the following values
                              {'mean', 'median', 'mode'}
            
    Outputs: 
            output - (DataFrame) dataset with imputed missing values 
            
    """
    num_data = data[list(num_cols)]
    if method == 'mode':
        output = num_data.fillna(getattr(num_data, method)().iloc[0])
    else:
        output = num_data.fillna(getattr(num_data, method)())
    return output