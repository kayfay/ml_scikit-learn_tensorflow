# Common imports
import os
import tarfile
import numpy as np

# Email and http handling imports
from six.moves import urllib
import email
import email.policy
import urlextract

# Data type handling imports
from collections import Counter

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Natural language processing imports
import nltk
import re
from html import unescape

# Fetch data
DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham_tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

# Load emails
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")

# Instance declerations
stemmer = nltk.PortStemmer()
url_extractor = urlextract.URLExtract()


# Funtion declarations


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL),
                          ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for single_email in emails:
        structure = get_email_structure(single_email)
        structures[structure] += 1
    return structure


def html_to_plain_text(html):
    # Preprocessing function
    # Using regex as opposed to the extra dependecy beautifulsoup
    text = re.sub('<head.*?>.*?</head>', '', html,
                  flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text,
                  flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text,
                  flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text,
                  flags=re.M | re.S)
    return unescape(text)


def email_to_text(email):
    # check email and convert to text based on content type
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


# Class declations


# Convert emails to word Counter data types
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for single_email in X:
            text = email_to_text(single_email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
            for url in urls:
                text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


# get data
fetch_spam_data()

# declare emails
ham_filenames = [name for name in
                 sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in
                  sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
ham_emails = [load_email(is_spam=False, filename=name)
              for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name)
               for name in spam_filenames]


# Check files lenghts
len(ham_filenames)   # 2500
len(spam_filenames)  # 500

# View an email
print(ham_emails[1].get_content().strip())

# view most common structures
structures_counter(ham_emails).most_common()   # note: 2400 plain
structures_counter(spam_emails).most_common()  # note: 183 html
# note: spam emails contain a large amount of html emails
# note: ham emails contain a larger amount of plain text emails

# Explore email headers
for header, value in spam_emails[0].items():
    print(header, ":", value)

# Note subject and from email address
spam_emails[0]["Subject"]   # Life Insurance - Why Pay More?
spam_emails[0]["From"]      # 12a1mailbot1@web.de

# Create test / train splits
X = np.array(ham_emails + spam_emails)
y = np.array([0] + len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
