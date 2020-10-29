__license__ = "Apache 2"
import argparse
import pathlib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import detail
import helpers

from sklearn.cluster import KMeans

abstract_md = pathlib.Path("doc/abstracts-df.md")
bert_sentence_md = pathlib.Path("doc/bert-sentence-desc.md")
kmeans_md = pathlib.Path("doc/kmeans-desc.md")

def _get_state(hash_funcs=None):
    session = helpers._get_session()
    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = helpers._SessionState(session, hash_funcs)
    return session._custom_session_state



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


def header_loading() -> tuple:
    state = _get_state()
    st.title("Stanford ETD Abstract Similarity")
    abstracts_df = pd.read_pickle("data/abstracts.pkl")
    with open("data/abstracts-bert-embeddings.pkl", "rb") as fo:
        abstracts_embeddings = pickle.load(fo)
    return abstracts_df, abstracts_embeddings


def about_dataframe_bert(dataframe: pd.DataFrame):
    st.markdown(abstract_md.read_text())
    st.write(dataframe.head())
    st.markdown(bert_sentence_md.read_text())


def about_kmeans():
    st.markdown(kmeans_md.read_text())
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


def cluster_results(main_content, cluster_size: int, clustered_abstracts: list):
    col1, col2 = main_content.beta_columns(2)
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
            if col1.button("Details", key=f"{cluster_size} {i}"):
                main_content.empty()
                detail.main(i, groups)
 
            for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
                col1.write(f"{dept} {len(druids)}")
        else:
            col2.subheader(header)
            if col2.button("Details", key=f"{cluster_size} {i}"):
                main_content.empty()
                detail.main(i, groups)
            for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
                col2.write(f"{dept} {len(druids)}")


def main():
    abstracts_df, abstracts_embeddings = header_loading()
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
        main_content = st.empty()
        cluster_results(main_content, option, clustered_abstracts)


if __name__ == "__main__":
#    streamlit run abstract_similarity.py 
    main()
