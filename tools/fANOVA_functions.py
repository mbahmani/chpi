import numpy as np
import pandas as pd
import time
import ConfigSpace
import ConfigSpace.hyperparameters as csh
import fanova
from sklearn.preprocessing import LabelEncoder


def do_fanova(dataset_name, algorithm, st=0, end=99):
    """
    Derive importance of hyperparameter combinations
    on the performance data for the given algorithm

    Input:
           csv_name - (DataFrame) contains the performance data
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees,
                         SVM, SVM_rbf, SVM_sigmoid, GradientBoosting}
           st - (int) starts from the specified number of dataset
           end - (int) ends at the specified number of dataset
    Output:
           writes the results on a csv file

    """
    data = pd.read_csv(dataset_name, low_memory=False)

    # define the config space
    # if SVM, then there are 3 cs's
    cs1, cs2 = config_space[algorithm]
    cols = col_names[algorithm]

    data = data.loc[:, cols]
    data.imputation.fillna('none', inplace=True)
    data = label_encoding(data, algorithm)
    datasets = data.dataset.unique()[st:end]
    results = pd.DataFrame()

    for indx, d_name in enumerate(datasets):
        print('Dataset {}({})'.format(indx + 1, d_name))
        selected = data.dataset == d_name
        data_filter = data.loc[selected, :]
        missing_values = sum(data_filter.imputation == 3) == 0

        try:
            df, time_taken = fanova_to_df(data_filter, algorithm,
                                          missing_values, cs1, cs2)

            df['dataset'] = d_name
            df['imputation'] = missing_values
            df['time_taken'] = time_taken

            results = pd.concat([results, df], axis=0)
            results.to_csv('performance_data/{}_fANOVA_results.csv'.format(algorithm),
                           header=True,
                           index=False)
        except Exception as e:
            print('***'
                  'The following error occured for {} dataset:{}'
                  '***'.format(d_name, e))


def fanova_to_df(data, algorithm, missing_values, cs1, cs2):
    """
    Derive importance of hyperparameter combinations
    for the given algorithm

    Input:
           data - (DataFrame) contains the performance data
                  for a dataset
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees,
                         SVM, GradientBoosting}
           missing_values - (boolean) whether imputation has
                            been done on the dataset
           cs1, cs2 - configuration space objects
    Output:
           df - (DataFrame) contains the variance contributions
                per hyperparameter combination
           time_taken - performance time in sec

    """
    if missing_values:
        X = data.loc[:, sorted(data.columns[1:-1])].values
        y = data.iloc[:, -1].values
        cs = cs1
    else:
        X = data.loc[:, sorted(data.columns[1:-2])].values
        y = data.iloc[:, -1].values
        cs = cs2

    f = fanova.fANOVA(X, y,
                      n_trees=32,
                      bootstrapping=True,
                      config_space=cs)

    start = time.perf_counter()
    print('Singles')
    imp1 = get_single_importance(f)
    print('Pairs')
    imp2 = f.get_most_important_pairwise_marginals()
    print('Triples')
    if missing_values:
        imp3_1 = get_triple_importance(f, algorithm)
        imp3_2 = get_triple_impute(f, algorithm)
        imp3 = dict_merge(imp3_1, imp3_2)
    else:
        imp3 = get_triple_importance(f, algorithm)

    imp = dict_merge(imp1, imp2, imp3)
    end = time.perf_counter()

    time_taken = end - start
    print('time taken is {} min'.format(time_taken / 60))

    df = pd.DataFrame({'param': list(imp.keys()),
                       'importance': list(imp.values())},
                      index=None)
    return df, time_taken

    """
    Defining the configuration space in case of
    Random Forest and Extra Trees Classifiers

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.CategoricalHyperparameter('bootstrap',
                                        choices=['0', '1'])
    hp2 = csh.CategoricalHyperparameter('criterion',
                                        choices=['0', '1'])
    hp3 = csh.CategoricalHyperparameter('imputation',
                                        choices=['0', '1', '2'])
    hp4 = csh.UniformFloatHyperparameter('max_features', lower=0.1,
                                         upper=0.9, log=False)
    hp5 = csh.UniformIntegerHyperparameter('min_samples_leaf', lower=1,
                                           upper=20, log=False)
    hp6 = csh.UniformIntegerHyperparameter('min_samples_split', lower=2,
                                           upper=20, log=False)
    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5, hp6])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp2, hp4, hp5, hp6])

    return cs1, cs2


def cs_km():
    """
    Defining the configuration space in case of
    kmeans

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.CategoricalHyperparameter('algorithm', choices=['0', '1', '2'])
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])
    hp3 = csh.CategoricalHyperparameter('init', choices=['0', '1'])

    hp4 = csh.UniformIntegerHyperparameter('n_clusters',
                                           lower=2, upper=40, log=False)
    hp5 = csh.UniformIntegerHyperparameter('max_iter',
                                           lower=50, upper=501, log=False)
    hp6 = csh.UniformIntegerHyperparameter('n_init',
                                           lower=2, upper=30, log=False)
    hp7 = csh.UniformFloatHyperparameter('tol', lower=0.00001,
                                         upper=0.1, log=False)
    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5, hp6, hp7])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5, hp6, hp7])

    return cs1, cs2


def cs_agg():
    """
    Defining the configuration space in case of
    agglomerativeclustering

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.CategoricalHyperparameter(
        'affinity', choices=['0', '1', '2', '3', '4'])
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])

    hp3 = csh.CategoricalHyperparameter(
        'linkage', choices=['0', '1', '2', '3'])
    hp4 = csh.CategoricalHyperparameter(
        'compute_full_tree', choices=['0', '1', '2'])
    hp5 = csh.UniformIntegerHyperparameter('n_clusters',
                                           lower=2, upper=40, log=False)
    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5])

    return cs1, cs2


def cs_affipro():
    """
    Defining the configuration space in case of
    agglomerativeclustering

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.UniformIntegerHyperparameter('max_iter',
                                           lower=50, upper=501, log=False)
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])

    hp3 = csh.UniformIntegerHyperparameter('convergence_iter',
                                           lower=2, upper=30, log=False)
    hp4 = csh.UniformFloatHyperparameter('damping', lower=0.5,
                                         upper=0.999, log=False)

    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, ])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, ])

    return cs1, cs2


def cs_spectral():
    """
    Defining the configuration space in case of
    agglomerativeclustering

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.UniformIntegerHyperparameter('n_clusters',
                                           lower=2, upper=40, log=False)
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])

    hp3 = csh.CategoricalHyperparameter('affinity', choices=['0', '1'])

    hp4 = csh.UniformIntegerHyperparameter('n_init',
                                           lower=2, upper=30, log=False)
    hp5 = csh.UniformIntegerHyperparameter('n_neighbors',
                                           lower=2, upper=40, log=False)

    hp6 = csh.CategoricalHyperparameter('assign_labels', choices=['0', '1'])

    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5, hp6])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5, hp6])

    return cs1, cs2


def cs_dbscan():
    """
    Defining the configuration space in case of
    agglomerativeclustering

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.UniformIntegerHyperparameter('min_samples',
                                           lower=2, upper=20, log=False)
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])

    hp3 = csh.UniformFloatHyperparameter('eps', lower=0.01,
                                         upper=1, log=False)
    hp4 = csh.CategoricalHyperparameter('metric', choices=['0', '1', '2', '3'])

    hp5 = csh.CategoricalHyperparameter(
        'algorithm', choices=['0', '1', '2', '3'])

    hp6 = csh.UniformIntegerHyperparameter('leaf_size',
                                           lower=2, upper=50, log=False)

    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5, hp6])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5, hp6])

    return cs1, cs2


def cs_meanshift():
    """
    Defining the configuration space in case of
    meanshift

    """
    cs1 = ConfigSpace.ConfigurationSpace()
    cs2 = ConfigSpace.ConfigurationSpace()

    hp1 = csh.UniformIntegerHyperparameter('max_iter',
                                           lower=50, upper=501, log=False)
    hp2 = csh.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])

    hp3 = csh.UniformFloatHyperparameter('bandwidth', lower=0.1,
                                         upper=2.5, log=False)
    hp4 = csh.CategoricalHyperparameter('bin_seeding', choices=['0', '1'])

    hp5 = csh.CategoricalHyperparameter('cluster_all', choices=['0', '1'])

    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5])

    return cs1, cs2


config_space = {'kmeans': cs_km(),
                "agglomerativeclustering": cs_agg(),
                "affinitypropagation": cs_affipro(),
                "spectralclustering": cs_spectral(),
                "dbscan": cs_dbscan(),
                "meanshift": cs_meanshift(),

                }

km_cols = ["dataset", "n_clusters", "init",
           "max_iter", "algorithm", "n_init", "tol",
           "imputation", 'normalized_mutual_info_score']

agg_cols = ["dataset", "n_clusters", "affinity",
            "linkage", "compute_full_tree",
            "imputation", 'normalized_mutual_info_score']

affipro_cols = ["dataset", "max_iter", "convergence_iter",
                "damping",
                "imputation", 'normalized_mutual_info_score']

spectral_cols = ["dataset", "n_clusters", "n_init",
                 "affinity", "n_neighbors", "assign_labels",
                 "imputation", 'normalized_mutual_info_score']

dbscan_cols = ["dataset", "eps", "min_samples",
               "metric", "algorithm", "leaf_size",
               "imputation", 'normalized_mutual_info_score']

meanshift_cols = ["dataset", "bin_seeding", "cluster_all",
                  "max_iter", "bandwidth",
                  "imputation", 'normalized_mutual_info_score']


col_names = {'kmeans': km_cols,
             'agglomerativeclustering': agg_cols,
             'affinitypropagation': affipro_cols,
             'spectralclustering': spectral_cols,
             'dbscan': dbscan_cols,
             'meanshift': meanshift_cols,

             }


def label_encoding(data, algorithm):
    """
    Performing label encoding for the categorical hyperparameters
    of the given algorithm

    Input:
           data - (DataFrame) contains the performance data
           algorithm - (str) takes one of the following options
                        {kmeans, agglomerativeclustering, affinitypropagation,
                         }
    Output:
           data - (DataFrame) contains only numerical features

    """
    le = LabelEncoder()

    if algorithm == 'kmeans':
        data.imputation = le.fit_transform(data.imputation)
        data.init = le.fit_transform(data.init)
        data.algorithm = le.fit_transform(data.algorithm)
    elif algorithm == 'agglomerativeclustering':
        data.imputation = le.fit_transform(data.imputation)
        data.affinity = le.fit_transform(data.affinity)
        data.linkage = le.fit_transform(data.linkage)
        data.compute_full_tree = le.fit_transform(data.compute_full_tree)

    elif algorithm == 'affinitypropagation':
        data.imputation = le.fit_transform(data.imputation)

    elif algorithm == 'spectralclustering':
        data.imputation = le.fit_transform(data.imputation)
        data.affinity = le.fit_transform(data.affinity)
        data.assign_labels = le.fit_transform(data.assign_labels)

    elif algorithm == 'dbscan':
        data.imputation = le.fit_transform(data.imputation)
        data.metric = le.fit_transform(data.metric)
        data.algorithm = le.fit_transform(data.algorithm)

    elif algorithm == 'meanshift':
        data.imputation = le.fit_transform(data.imputation)
        data.bin_seeding = le.fit_transform(data.bin_seeding)
        data.cluster_all = le.fit_transform(data.cluster_all)

    return data


def get_single_importance(f):
    """
    Derive importance of each hyperparameter

    Input:
           f - (fANOVA) object
    Output:
           imp1 - (dict) key: hyperparameter name
                         value: variance contribution

    """
    names = f.cs.get_hyperparameter_names()

    imp1 = {}
    for name in names:
        imp1_ind = f.quantify_importance([name])
        value = imp1_ind[(name,)]['individual importance']
        imp = {name: value}
        imp1.update(imp)

    return imp1


def get_importance(f, *params):
    """
    Derive importance of the specified
    combination of hyperparameters
    Input:
           f - (fANOVA) object
           *params - (str) names of hyperparameters
    Output:
           imp - (dict) key: hyperparameter combination
                         value: variance contribution

    """
    imp = f.quantify_importance(list(params))
    value = imp[params]['individual importance']
    imp = {params: value}
    return imp


def dict_merge(*args):
    """
    Merges several python dictionaries

    Input:
           *args - (dict) python dictionaries
    Output:
           imp - (dict) merged dictionary

    """
    imp = {}
    for dictt in args:
        imp.update(dictt)
    return imp


def get_triple_importance(f, algorithm):
    """
    Derive importance of specified triple combinations
    of hyperparameters per algorithm

    Input:
           f - (fANOVA) object
           algorithm - (str) takes one of the following options
                        {kmeans, agglomerativeclustering, affinitypropagation,
                         SVM}
    Output:
           imp - (dict) key: hyperparameter name
                        value: variance contribution

    """
    if algorithm == 'kmeans':
        imp1 = get_importance(f, 'n_clusters',
                              'init', 'max_iter')
        imp2 = get_importance(f, 'n_clusters',
                              'init', 'algorithm')
        imp = dict_merge(imp1, imp2)
    elif algorithm == 'agglomerativeclustering':
        imp1 = get_importance(f, 'n_clusters', 'affinity', 'linkage')
        imp2 = get_importance(f, 'compute_full_tree',
                              'affinity', 'linkage')
        imp = dict_merge(imp1, imp2)

    elif algorithm == 'affinitypropagation':
        imp = get_importance(f, 'max_iter', 'convergence_iter', 'damping')

    elif algorithm == 'spectralclustering':
        imp1 = get_importance(f, 'n_clusters', 'n_init', 'affinity')
        imp2 = get_importance(f, 'n_clusters',
                              'assign_labels', 'n_neighbors')
        imp = dict_merge(imp1, imp2)

    elif algorithm == 'dbscan':
        imp1 = get_importance(f, 'eps', 'min_samples', 'metric')
        imp2 = get_importance(f, 'eps',
                              'algorithm', 'leaf_size')

        imp = dict_merge(imp1, imp2)

    elif algorithm == 'meanshift':
        imp1 = get_importance(f, 'bin_seeding', 'cluster_all', 'max_iter')
        imp2 = get_importance(f, 'cluster_all',
                              'max_iter', 'bandwidth')

        imp = dict_merge(imp1, imp2)
    return imp


def get_triple_impute(f, algorithm):
    """
    Derive importance of specified triple combinations
    of hyperparameters per algorithm in case of
    data imputation

    Input:
           f - (fANOVA) object
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees,
                         SVM, GradientBoosting}
    Output:
           imp - (dict) key: hyperparameter name
                        value: variance contribution

    """
    if np.isin(algorithm, ["kmeans"]):
        imp = get_importance(f, 'imputation',
                             'n_clusters', 'algorithm')
    elif algorithm == 'agglomerativeclustering':
        imp = get_importance(f, 'imputation', 'n_clusters', 'affinity')

    elif algorithm == 'affinitypropagation':
        imp = get_importance(f, 'imputation', 'damping', 'convergence_iter')

    elif algorithm == 'spectralclustering':
        imp = get_importance(f, 'imputation', 'n_clusters', 'n_init')

    elif algorithm == 'dbscan':
        imp = get_importance(f, 'imputation', 'eps', 'min_samples')

    elif algorithm == 'meanshift':
        imp = get_importance(f, 'imputation', 'max_iter', 'cluster_all')

    return imp
