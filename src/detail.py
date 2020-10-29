__license__ = "Apache 2"

import helpers
import pandas as pd
import streamlit as st
import spacy

from spacy import displacy


abstracts_df = pd.read_pickle("data/abstracts.pkl")
setup_spacy = st.cache(helpers.setup_spacy, allow_output_mutation=True)


def main(cluster_number: int, groups: dict) -> None:
    nlp, fast_vocab = setup_spacy()
    st.subheader(f"Details for Cluster {cluster_number+1}")
    fast_suggestions = st.empty()
    fast_subjects = dict()
    for dept, druids in reversed(
                sorted(groups.items(), key=lambda item: len(item[1]))
            ):
        st.markdown(f"### {dept} Total {len(druids)}")
        for druid in druids:
            abstract = abstracts_df.loc[abstracts_df['druids'] == druid]
            st.markdown(f"#### {druid} [{abstract['title'].item()}](https://purl.stanford.edu/{druid})")
            st.markdown("#### FAST Suggestions")
            doc = nlp(abstract['abstracts_cleaned'].item())
            for doc_entity in doc.ents:
                fast_uri = fast_vocab.keyword_processor.get_keyword(doc_entity.text)
                if fast_uri in fast_subjects:
                    fast_subjects[fast_uri]['weight'] += 1
                else:
                    series = helpers.FAST_VOCAB.loc[helpers.FAST_VOCAB['URI'] == fast_uri]
                    if len(series) < 1:
                        print(f"Zero length, {fast_uri} {doc_entity.text}")
                    fast_subjects[fast_uri] = { 'weight': 1,
                                                'label': series['Label'].item() }
            st.write(abstract['abstracts'].item())
  
    fast_suggestions.write(fast_subjects)

             
