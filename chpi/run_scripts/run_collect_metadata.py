from tools.functions import clusering_per_algorithm
import pandas as pd


def run_experiment():
    # "dbscan", "affinitypropagation", "agglomerativeclustering", "Kmeans", "meanshift", "spectralclustering"
    for cls in {"optics"}:

        clusering_per_algorithm(path="datasets/", algorithm=cls)

    return "Done!"


def df_edit():
    for cls in {"Kmeans"}:
        df = pd.read_csv("./performance_data/"+cls+"_results.csv")
        df.loc[df['imputation'] == "None", 'imputation'] = ""
        df.to_csv("./performance_data/"+cls+"_results.csv", index=False)


if __name__ == '__main__':

    # run_marginal_contribution()
    run_experiment()
