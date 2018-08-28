import os
import urlextract
from six.moves import urllib

# Fetch data
DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"

path = os.path.join("datasets", "spam")
urllib.request.urlretrieve(HAM_URL, path + 'f.tar.bz2')  

urllib.request.urlretrieve(SPAM_URL, path + 'f2.tar.bz2')  

