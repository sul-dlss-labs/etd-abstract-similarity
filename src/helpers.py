__license__ = "Apache 2"

import pathlib
import pickle
import re


import numpy as np
import pandas as pd

import spacy

import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from spacy_lookup import Entity
from streamlit.report_thread import get_report_ctx
from streamlit.hashing import _CodeHasher
from streamlit.server.server import Server


ABSTRACT_MD = pathlib.Path("doc/abstracts-df.md")
BERT_SENTENCE_MD = pathlib.Path("doc/bert-sentence-desc.md")
FAST_GEO_DF = pd.read_json("data/fast-geo.json")
FAST_NER_MD = pathlib.Path("doc/fast-ner-spacy.md")
FAST_TOPICS_DF = pd.read_json("data/fast-topics.json")
FAST_VOCAB = pd.concat([FAST_GEO_DF, FAST_TOPICS_DF])
KMEANS_MD = pathlib.Path("doc/kmeans-desc.md")
SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z]')
STOP_WORDS_LIST = stopwords.words('english')



@st.cache(allow_output_mutation=True)
def get_cluster(dataframe: pd.DataFrame,
                embeddings: np.ndarray,
                n_clusters=30) -> np.ndarray:
    clustering_model = KMeans(n_clusters=n_clusters)
    predicted_cluster = clustering_model.fit_predict(embeddings)
    clustered_abstracts = dataframe.copy()
    clustered_abstracts['cluster'] = clustered_abstracts.index.map(
        lambda x: predicted_cluster[x])
    groups = [group for group in clustered_abstracts.groupby(by='cluster')]
    return groups





def about_dataframe_bert(dataframe: pd.DataFrame):
    st.markdown(ABSTRACT_MD.read_text())
    st.write(dataframe.head())
    st.markdown(BERT_SENTENCE_MD.read_text())

def cluster_sizes_graph():
    with open("data/wcss.pkl", "rb") as fo:
        wcss = pickle.load(fo)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 50), wcss, marker="o", linestyle="--")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("K-means Clustering")
    st.pyplot(plt)

def about_kmeans():
    st.markdown(KMEANS_MD.read_text())
    cluster_sizes_graph()

def about_fast_ner():
    st.markdown(FAST_NER_MD.read_text())
    st.write(FAST_TOPICS_DF.sample(10))

class _SessionState:
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to
        fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                    self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def save_fast_to_druid(druid: str, fast_uris: list):
    if len(fast_uris) < 1:
        return


def cleanup(term: str) -> str:
    cleaned = []
    for char in term.split():
        cleaned_char = SPECIAL_CHAR_RE.sub(' ', char).lower()
        if cleaned_char in STOP_WORDS_LIST:
            continue
        cleaned.append(cleaned_char)
    return ' '.join(cleaned)


def generate_labels(df: pd.DataFrame) -> dict:
    labels = {}
    for row in df.iterrows():
        uri = row[1]["URI"]
        label = row[1]["cleaned"]
        labels[uri] = [label, ]
    return labels


def load_fast(fast_df: pd.DataFrame, label: str) -> tuple:
    fast = spacy.load("en_core_web_sm")
    fast_labels = generate_labels(fast_df)
    fast_entity = Entity(keywords_dict=fast_labels, label=label)
    fast.add_pipe(fast_entity)
    fast.remove_pipe("ner")
    return fast, fast_entity


# Setup FAST Topic, Geographical, and Chronological Spacy Vocabularies
def setup_spacy() -> tuple:
    # Loads Topics and Geographic to Spacy Vocabularies
    geo_nlp, geo_entity = load_fast(FAST_GEO_DF, 'Geo')
    topic_nlp, topic_entity = load_fast(FAST_TOPICS_DF, 'Topic')
    return geo_nlp, geo_entity, topic_nlp, topic_entity
