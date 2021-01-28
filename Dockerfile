FROM python:3.8-alpine

WORKDIR /usr/src/app

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev g++ git \
    libffi-dev openssl-dev \
    libxml2 libxml2-dev \
    libxslt libxslt-dev \
    libjpeg-turbo-dev zlib-dev

COPY . .

RUN pip install --no-cache -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

EXPOSE 5000

CMD streamlit run src/app.py
