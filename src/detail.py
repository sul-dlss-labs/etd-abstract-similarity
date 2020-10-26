__license__ = "Apache 2"

import streamlit as st
import pandas as pd

abstracts_df = pd.read_pickle("data/abstracts.pkl")

def main(cluster_number: int, groups: dict) -> None:
    st.subheader(f"Details for Cluster {cluster_number+1}")
    for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
        st.markdown(f"### {dept} Total {len(druids)}")
        for druid in druids:
            abstract = abstracts_df.loc[abstracts_df['druids'] == druid]
            st.markdown(f"#### {druid} [{abstract['title'].item()}](https://purl.stanford.edu/{druid})")
            st.write(abstract['abstracts'].item())
