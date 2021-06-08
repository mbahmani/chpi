from tools.functions import clusering_per_algorithm,clustering_per_dataset


if __name__ == '__main__':
    path_to_datasets="datasets"
    clusering_per_algorithm(path=path_to_datasets, algorithm="kmeans")