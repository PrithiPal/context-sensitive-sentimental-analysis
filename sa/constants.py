import socialsent_util
import os
from nltk.corpus import stopwords
from pkg_resources import resource_filename

DATA = os.path.abspath('embeddings_socialsent') + "/"
print('data = ',DATA)
PROCESSED_LEXICONS = 'lexicons_socialsent/'
POLARITIES = DATA + 'polarities/'
STOPWORDS = set(stopwords.words('english'))
LEXICON = 'inquirer'
YEARS = map(str, range(1850, 2000, 10))


GOOGLE_EMBEDDINGS = './word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_EMBEDDINGS = './example_embeddings/glove.6B.100d.txt'
YELP_EMBEDDINGS = './example_embeddings/yelp_w2v_pure'




