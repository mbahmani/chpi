import pandas as pd
from tools.helper import obtain_marginal_contributions, marginal_plots, determine_relevant


def draw_plots():
    # draw plots based on fanova resutls
    # "dbscan", "optics", "agglomerativeclustering", "kmeans", "meanshift", "spectralclustering"
    for cls in {"dbscan"}:
        # "normalized_mutual_info_score", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "dunn_score", "c_index_score", "ratkowsky_lance_score", "mc_clain_rao_score"
        for metric in {"normalized_mutual_info_score", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "dunn_score", "c_index_score", "ratkowsky_lance_score", "mc_clain_rao_score"}:
            df = pd.read_csv("performance_data/"+cls+"_"+metric+"_fANOVA_results.csv")
            total_ranks, marginal_contribution, _ = obtain_marginal_contributions(df)
            sorted_values, keys = determine_relevant(marginal_contribution, max_interactions=0)
            marginal_plots(sorted_values, keys, cls+"_"+metric)
    return "Done!"


if __name__ == '__main__':

    try:
        print(draw_plots())
    except Exception as e:
        print(e.message)
