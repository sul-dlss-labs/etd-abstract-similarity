# Electronic Theses and Dissertations (ETD)
## Abstract Similarity

Live application on Heroku at https://etd-abstract-similarity.herokuapp.com/.

## Using Docker
To run this application using the [Docker](https://www.docker.com) image,
`docker run -p 5000:5000 suldlss/etd-abstract`. The application will be running
on port 5000 as **http://localhost:5000**.

## Development
1. Have [Python](https://python.org) 3.8 or above installed.
1. Clone the  source code repository  
`git clone https://github.com/sul-dlss-labs/etd-abstract-similarity.git .`.
1. Install supporting python modules `pip install -r requirements.txt`
1. Download and install [NTLK]() stopwords `python -m nltk.downloader stopwords`
1. Download and install [spaCy models](https://spacy.io/models)  
   `python -m spacy download en_core_web_sm`.

Now, you can run the Streamlit with `streamlit run src/app.py` and connect with
your local browser  **http://localhost:5000**.
