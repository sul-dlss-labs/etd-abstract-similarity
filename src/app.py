__license__ = "Apache 2"
import argparse
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="AI-ETD Cataloging Utility")
parser.add_argument("--biology", default=False)

abstracts_df_desc = """## Abstracts DataFrame
For all of the world-accessible theses and dissertations, we queried Stanford's
Digital Repository with value in *druids* column and retrieved the abstract
contained in the MODS XML data-stream.  Each abstract was then processed to
remove stop words and punctuation marks, converted to lower case, and stored
in the *abstracts_cleaned* column. Finally, the department name was retrieved
from the  MODS and added as the *department* column.
"""

BERT_sentence_desc = """## BERT Sentence Embeddings
Starting with the *abstracts_cleaned* column, we use a pre-trained BERT model
for creating abstract embedding with the Huggingface
[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
library.

```python
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
abstracts_embeddings = sbert_model.encode(abstracts_df['abstracts_cleaned'])
```
"""

kmeans_desc = """## K-Means Clusters
Using the `abstracts_embeddings` matrix, the K-Means clustering approach takes
a number of clusters and
centers each cluster by calculating the Euclidean distance between the center
in each cluster with its observations.

One method to determine the number of clusters is to calculate the
**Within Cluster Sum of Squares (wcss)** for a range of cluster sizes. The
*wcss* is the sum of variance between the observations in each cluster.
Graphing the *wcss* over the number of clusters, we can pick a number that
balances the steepness of the curve verses the smoothing tail.

For more information, refer to
[K-means clustering](https://365datascience.com/k-means-clustering/)  article.
"""


@st.cache(allow_output_mutation=True)
def get_cluster(dataframe, embeddings, n_clusters=30):
    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(embeddings)
    clustering_abstracts = [[] for i in range(n_clusters)]
    for abstract_id, cluster_id in enumerate(clustering_model.labels_):
        series = dataframe.iloc[abstract_id]
        clustering_abstracts[cluster_id].append(
            {"druid": series["druids"], "department": series["departments"]}
        )
    return clustering_abstracts


def header_loading(biology_only):
    title = "Stanford ETD Abstract Similarity"
    if biology_only:
        title = f"{title} for Biology"
    st.title(title)
    loading_text = st.text("Loading Abstracts and Encoding with BERT...")
    if biology_only:
        abstracts_df = pd.read_pickle("../data/biology.pkl")
        with open("data/biology-bert-embedding.pkl", "rb") as fo:
            abstracts_embeddings = pickle.load(fo)
    else:
        abstracts_df = pd.read_pickle("data/abstracts.pkl")
        with open("data/abstracts-bert-embeddings.pkl", "rb") as fo:
            abstracts_embeddings = pickle.load(fo)
    loading_text.text("Finished loading")
    return abstracts_df, abstracts_embeddings


def about_dataframe_bert(dataframe: pd.DataFrame):
    st.markdown(abstracts_df_desc)
    st.write(dataframe.head())
    st.markdown(BERT_sentence_desc)


def about_kmeans():
    st.markdown(kmeans_desc)
    cluster_sizes_graph()


def cluster_sizes_graph():
    with open("data/wcss.pkl", "rb") as fo:
        wcss = pickle.load(fo)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 50), wcss, marker="o", linestyle="--")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("K-means Clustering")
    st.pyplot(plt)


def cluster_results(clustered_abstracts):
    col1, col2 = st.beta_columns(2)
    for i, cluster in enumerate(clustered_abstracts):
        header = f"Cluster {i+1} size: {len(cluster)}"
        groups = {}
        for abstract in cluster:
            dept = abstract["department"].replace(".", "")
            druid = abstract["druid"]
            if dept in groups:
                groups[dept].append(druid)
            else:
                groups[dept] = [
                    druid,
                ]
        if not i % 2:
            col1.subheader(header)
            for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
                col1.write(f"{dept} {len(druids)}")
        else:
            col2.subheader(header)
            for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
                col2.write(f"{dept} {len(druids)}")


def main(biology_only):
    abstracts_df, abstracts_embeddings = header_loading(biology_only)
    abstract_btn = st.sidebar.button("About Abstracts and BERT")
    kmeans_btn = st.sidebar.button("About KMeans Clustering")
    st.sidebar.button("Reset")
    if abstract_btn:
        about_dataframe_bert(abstracts_df)
    if kmeans_btn:
        about_kmeans()
    if not abstract_btn and not kmeans_btn:
        option = st.slider("Number of Clusters", 1, 50)
        st.header(f"Cluster size is {option}")
        clustered_abstracts = get_cluster(abstracts_df,
                                          abstracts_embeddings,
                                          option)
        cluster_results(clustered_abstracts)


if __name__ == "__main__":
    print(
        """
To run with just the biology abstract:
streamlit run abstract_similarity.py -- --biology True
"""
    )
    args = parser.parse_args()
    main(args.biology)
