import socialsent_util
import os
from nltk.corpus import stopwords
from pkg_resources import resource_filename

### SYSTEM AGNOSTIC CONSTANTS 
######
DATA = os.path.abspath('embeddings_socialsent') + "/"
print('data = ',DATA)
PROCESSED_LEXICONS = 'lexicons_socialsent/'
POLARITIES = DATA + 'polarities/'
STOPWORDS = set(stopwords.words('english'))
LEXICON = 'inquirer'
YEARS = map(str, range(1850, 2000, 10))

######
## THE FOLLOWING CAN BE REPLACED BY DOWNLOADING APPROPRIATE RESOURCES AND CHANGING PATHS:

## GOOGLE EMBEDDING DOWNLOADED

GOOGLE_EMBEDDINGS = './word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_EMBEDDINGS = './example_embeddings/glove.6B.100d.txt'



#from http://www.cis.lmu.de/~sascha/Ultradense/
TWITTER_EMBEDDINGS = '/dfs/scratch0/gigawordvecs/twitter_lower_cw1_sg400_transformed.txt'



# The following can be constructed from the corpora cited in the paper
STOCK_EMBEDDINGS = '/lfs/madmax9/0/stock/svd-vecs'
STOCK_COUNTS = '/lfs/madmax3/0/stock/vecs.bin'
COHA_EMBEDDINGS = '/dfs/scratch0/COHA/cooccurs/word/ppmi/lsmooth0/nFalse/neg1/cdsTrue/svd/300/50000/'
COHA_PPMI = '/dfs/scratch0/COHA/cooccurs/word/ppmi/lsmooth0/nFalse/neg1/cdsTrue/'
COHA_COUNTS = '/dfs/scratch0/COHA/cooccurs/word/4/'
COHA_SGNS_EMBEDDINGS = '/dfs/scratch0/COHA/cooccurs/word/sgns/300/'
FREQS = "/dfs/scratch0/hist_words/coha-word/freqs.pkl"
COHA_FREQS = "/dfs/scratch0/COHA/decade_freqs/{}-word.pkl"
DFS_DATA = '/dfs/scratch0/googlengrams/eng-all/decades/'
#POS = DFS_DATA + '/pos/'
POS = "/dfs/scratch0/hist_words/coha-word/pos/"
SUBREDDIT_EMBEDDINGS = '/dfs/scratch2/wleif/Reddit/vecs/{}/vecs'

def make_directories():
    socialsent_util.mkdir(DATA)
    socialsent_util.mkdir(PROCESSED_LEXICONS)
    socialsent_util.mkdir(POLARITIES)

make_directories()
