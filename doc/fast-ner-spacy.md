## Using FAST Vocabulary as Named Entity Recognition (NER)
Initially used in the [Biology Abstract](https://biology-fast-etds.herokuapp.com/) 
Streamlit application, we decided to create a [spaCy][SPACY] NER 
pipeline with over 460,000 FAST topic entities. Inspired by SUL Species 
Occurrence pilot, we created a new NER pipeline that extracted the URLs 
and Labels from the FAST
topic entities that allows us to tag matched terms in an ETD abstract.

```python
def load_fast(fast_df: pd.DataFrame, label: str) -> tuple:
    fast = spacy.load("en_core_web_sm")
    fast_labels = generate_labels(fast_df)
    fast_entity = Entity(keywords_dict=fast_labels, label=label)
    fast.add_pipe(fast_entity)
    fast.remove_pipe("ner")
    return fast, fast_entity
```

### Example of 10 random Topic entities: 


[SPACY]: https://spacy.io/
