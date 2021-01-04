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

st.set_page_config(page_title='ETD Abstract Similarity')

def _get_state(hash_funcs=None):
    session = helpers._get_session()
    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = helpers._SessionState(session,
                                                              hash_funcs)
    return session._custom_session_state

@st.cache(allow_output_mutation=True)
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
   

def loading() -> tuple:
    
    #st.image("src/sul-logo.svg")
    abstracts_df = pd.read_pickle("data/abstracts.pkl")
    with open("data/abstracts-bert-embeddings.pkl", "rb") as fo:
        abstracts_embeddings = pickle.load(fo)
    return abstracts_df, abstracts_embeddings 



#def cluster_results(cluster_size: int,
#                    clustered_abstracts: list) -> None:


def main():
    head = st.empty() 
    head.title("Stanford ETD Abstract Similarity")
    abstracts_df, abstracts_embeddings = loading()
    abstract_btn = st.sidebar.button("Abstracts and BERT")
    kmeans_btn = st.sidebar.button("KMeans Clustering")
    spacy_btn = st.sidebar.button("FAST NER")
    st.sidebar.button("Reset")
    if abstract_btn:
        helpers.about_dataframe_bert(abstracts_df)
    if kmeans_btn:
        helpers.about_kmeans()
    if spacy_btn:
        helpers.about_fast_ner()
    if not abstract_btn and not kmeans_btn and not spacy_btn:
        main_content = st.beta_container()

        option = main_content.slider("Number of Clusters", 1, 50)
        main_content.header(f"Cluster size is {option}")
        clustered_abstracts = helpers.get_cluster(abstracts_df,
                                                  abstracts_embeddings,
                                                  option)
        for row in clustered_abstracts:
            number = row[0]
            header = f"Cluster {number+1} size: {len(row[1])}"
            areas = [area for area in row[1].groupby(by='area')]
            main_content.subheader(header)
            if main_content.button("Tag", key=f"{option} {number}"):
                detail.main(main_content, number, areas)
            if main_content.button("DataFrame", key=f"df-{option}-{number}"):
                for area in areas:
                    main_content.write(f"{area[0]} {len(area[1])}")
                    main_content.dataframe(area[1])
           #st.markdown(get_json_download(row[1]), unsafe_allow_html=True)         
            ax = areas_in_cluster_graph(row[1])
            ax.set_title(f"Number of Druids by Area\nCluster {number+1} of {option}")
            main_content.pyplot(plt)



if __name__ == "__main__":
    main()
