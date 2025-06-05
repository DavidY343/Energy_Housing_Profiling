import pandas as pd


class UnifyClusters:
    def __init__(self, kmeans_path='dataset/cluster_cups_kmeans.csv',
                 bkmeans_path='dataset/cluster_cups_bkmeans.csv',
                 sc_path='dataset/cluster_cups_sc.csv',
                 output_path='dataset/clusters_unificado.csv'):
        self.kmeans_path = kmeans_path
        self.bkmeans_path = bkmeans_path
        self.sc_path = sc_path
        self.output_path = output_path

    def unify(self):
        df_kmeans = pd.read_csv(self.kmeans_path)
        df_bkmeans = pd.read_csv(self.bkmeans_path)
        df_sc = pd.read_csv(self.sc_path)

        df_kmeans = df_kmeans.rename(columns={'CLUSTER': 'CLUSTER_KMEANS'})
        df_bkmeans = df_bkmeans.rename(columns={'CLUSTER': 'CLUSTER_BKMEANS'})
        df_sc = df_sc.rename(columns={'CLUSTER': 'CLUSTER_SC'})

        df_unificado = df_kmeans.merge(df_bkmeans, on='CUPS').merge(df_sc, on='CUPS')
        df_unificado.to_csv(self.output_path, index=False)