__license__ = "Apache 2"

import helpers
import pandas as pd
import streamlit as st
import spacy

from spacy import displacy

setup_spacy = st.cache(helpers.setup_spacy, allow_output_mutation=True)
abstracts_df = pd.read_pickle("data/abstracts.pkl")

def main(container, cluster_number, groups) -> None:
    #geo_nlp, geo_entity, topic_nlp, topic_entity = setup_spacy()
    container.empty()
    container.header(f"Details for Cluster {cluster_number+1}")
    fast_suggestions = st.empty()
    fast_subjects = dict()
    #for group in groups:
    #    st.subheader(f"{group[0]} {len(group[1])}")
    #    st.write(group[1])
    for dept, druids in reversed(
                 sorted(groups, key=lambda item: len(item[1]))
            ):
         container.subheader(f"{dept} Total {len(druids)}")
         for druid in druids.iterrows():
             container.subheader(druid[1]['title'])
             abstract = druid[1]['abstracts']
             container.write(abstract)
    #         st.markdown(f"#### {druid} [{abstract['title'].item()}](https://purl.stanford.edu/{druid})")
    #         st.markdown("#### FAST Suggestions")
    #         doc = nlp(abstract['abstracts_cleaned'].item())
    #         for doc_entity in doc.ents:
    #             fast_uri = fast_vocab.keyword_processor.get_keyword(doc_entity.text)
    #             if fast_uri in fast_subjects:
    #                 fast_subjects[fast_uri]['weight'] += 1
    #             else:
    #                 series = helpers.FAST_VOCAB.loc[helpers.FAST_VOCAB['URI'] == fast_uri]
    #                 if len(series) < 1:
    #                     print(f"Zero length, {fast_uri} {doc_entity.text}")
    #                 fast_subjects[fast_uri] = { 'weight': 1,
    #                                             'label': series['Label'].item() }
    #         st.write(druid)
    #
    #fast_suggestions.write(fast_subjects)
