__license__ = "Apache 2"
import re

import requests

import lxml.etree as etree

from nltk.corpus import stopwords

NS = {"mods": "http://www.loc.gov/mods/v3"}
SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z]')
STOPWORDS = stopwords.words('english')

def cleanup(term: str) -> str:
    cleaned = []
    for char in term.split():
        cleaned_char = SPECIAL_CHAR_RE.sub(' ', char).lower()
        if cleaned_char in STOPWORDS:
            continue
        cleaned.append(cleaned_char)
    return ' '.join(cleaned)


def get_abstract(purl_xml: etree.XML) -> str:
    abstract = purl_xml.find("mods:abstract", namespaces=NS)
    if abstract is not None:
        return abstract.text


def get_department(purl_xml: etree.XML) -> str:
    corporate_names = purl_xml.findall(
        "mods:name[@type='corporate']/mods:namePart",
        namespaces=NS)
    department = []
    for name in corporate_names:
        if name.text.startswith("Stanford University"):
            continue
        department.append(name.text)
    return ','.join(department)


def get_mods(druid: str) -> etree.XML:
    purl_url = f"https://purl.stanford.edu/{druid}.mods"
    try:
        purl_result = requests.get(purl_url)
    except:
        print(f"Request error for {druid} with {purl_url}")
        return ''
    try:
        purl_xml = etree.XML(purl_result.text.encode())
    except etree.XMLSyntaxError:
        print(f"\nXML Syntax Error for {druid}")
        return ''
    return purl_xml

def get_title(purl_xml: etree.XML) -> str:
    title = purl_xml.find("mods:titleInfo/mods:title", namespaces=NS)
    if title is not None:
        return title.text
