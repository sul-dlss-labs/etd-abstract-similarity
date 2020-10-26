## BERT Sentence Embeddings
Starting with the *abstracts_cleaned* column, we use a pre-trained BERT model
for creating abstract embedding with the Huggingface
[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
library.

```python
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
abstracts_embeddings = sbert_model.encode(abstracts_df['abstracts_cleaned'])
```
