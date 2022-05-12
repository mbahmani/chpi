from tools.fANOVA_functions import do_fanova


def run():
    # do fanova
    for cls in {"dbscan", "optics", "agglomerativeclustering", "Kmeans", "meanshift", "spectralclustering"}:
        # "normalized_mutual_info_score","silhouette_score","davies_bouldin_score","calinski_harabasz_score","dunn_score","c_index_score","ratkowsky_lance_score","mc_clain_rao_score"
        for met in {"ratkowsky_lance_score"}:
            do_fanova(dataset_name="performance_data/"+cls+"_results.csv", algorithm=cls, metric=[met], st=0,end=99)

    return "Done!"


if __name__ == '__main__':
    try:
        print(run())
    except Exception as e:
        print(e.message)
