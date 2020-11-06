__license__ = "Apache 2"

import base64
import pathlib
import pickle


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import streamlit as st
import detail
import helpers

def _get_state(hash_funcs=None):
    session = helpers._get_session()
    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = helpers._SessionState(session,
                                                              hash_funcs)
    return session._custom_session_state


def areas_in_cluster_graph(cluster_df: pd.DataFrame):
    #! Should use df.plot instead of plt directly to generate figure
    area_data = {}
    for row in cluster_df.iterrows():
        area = row[1]['area']
        if area in area_data:
            area_data[area] += 1
        else:
            area_data[area] = 1
    sorted_areas = sorted(area_data.items(), key=lambda x: x[1])
    y_labels = [x[0] for x in sorted_areas]
    values = [i[1] for i in sorted_areas]
    fig, ax = plt.subplots(figsize=(10, 15))
    y_pos = np.arange(len(area_data))
    ax.barh(y_pos, values, color='#d1e6f0')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, x=.02, ha='left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    max_size = values[-1] 
    for i, v in enumerate(values):
        ax.text(max_size, i, str(v), ha='left', va='center')
    ax.set_xlabel('Number of Druids')
    # ax.set_title(f"Cluster ")
    return ax
   

def header_loading() -> tuple:
    st.title("Stanford ETD Abstract Similarity")
    #st.image("src/sul-logo.svg")
    abstracts_df = pd.read_pickle("data/abstracts.pkl")
    with open("data/abstracts-bert-embeddings.pkl", "rb") as fo:
        abstracts_embeddings = pickle.load(fo)
    return abstracts_df, abstracts_embeddings 



def cluster_results(cluster_size: int,
                    clustered_abstracts: list) -> None:
    for row in clustered_abstracts:
        number = row[0]
        header = f"Cluster {number+1} size: {len(row[1])}"
        areas = [area for area in row[1].groupby(by='area')]
        st.subheader(header)
        if st.button("Details", key=f"{cluster_size} {number}"):
            detail.main(number, areas)
        # st.markdown(get_json_download(row[1]), unsafe_allow_html=True)         
        ax = areas_in_cluster_graph(row[1])
        ax.set_title(f"Number of Druids by Area\nCluster {number+1} of {cluster_size}")
        st.pyplot(plt)

def get_json_download(df):
    json_str = df.to_json()
    b64 = base64.b64encode(json_str.encode())
    return f'<a href="data:file/json;base64,{b64}">Download json</a>'

def main():
    abstracts_df, abstracts_embeddings = header_loading()
    abstract_btn = st.sidebar.button("About Abstracts and BERT")
    kmeans_btn = st.sidebar.button("About KMeans Clustering")
    st.sidebar.button("Reset")
    if abstract_btn:
        helpers.about_dataframe_bert(abstracts_df)
    if kmeans_btn:
        helpers.about_kmeans()
    if not abstract_btn and not kmeans_btn:
        option = st.slider("Number of Clusters", 1, 50)
        st.header(f"Cluster size is {option}")
        clustered_abstracts = helpers.get_cluster(abstracts_df,
                                                  abstracts_embeddings,
                                                  option)
        cluster_results(option, clustered_abstracts)


if __name__ == "__main__":
    main()
