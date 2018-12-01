import pandas as pd
import numpy as np
from afinn import Afinn
import nltk
from nltk.corpus import sentiwordnet as swn
from normalization import normalize_accented_characters, html_parser, strip_html
from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report
import sys 
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')

### PREPARING THE MOVIE DATASET
def prepare_movie_dataset(train_start,train_end,test_start,test_end) : 

    
    dataset = pd.read_csv(r'datasets/movie_reviews.csv')
    print('dataset size : ',dataset.shape[0])

    train_data = dataset[train_start:train_end]
    test_data = dataset[test_start:test_end]
    
    print('Train_X : ',train_data.shape[0])
    print('Test_X  : ',test_data.shape[0])

    test_reviews = np.array(test_data['review'])
    test_sentiments = np.array(test_data['sentiment'])

    return train_data,test_reviews,test_sentiments

### GIVES SENTENCE POLARITY WITH SUMMATION OVER WORDNET LEXICON SCORES.
def analyze_sentiment_sentiwordnet_lexicon(review,
                                           verbose=False):
    
    review = html_parser.unescape(review)
    review = strip_html(review)
    
    text_tokens = nltk.word_tokenize(review)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0

    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and swn.senti_synsets(word, 'n'):
            ss_set = list(swn.senti_synsets(word, 'n'))
            if ss_set : 
                ss_set=ss_set[0]
        elif 'VB' in tag and swn.senti_synsets(word, 'v'):
            ss_set = list(swn.senti_synsets(word, 'v'))
            if ss_set : 
                ss_set=ss_set[0]
        elif 'JJ' in tag and swn.senti_synsets(word, 'a'):
            ss_set = list(swn.senti_synsets(word, 'a'))
            if ss_set : 
                ss_set=ss_set[0]
        elif 'RB' in tag and swn.senti_synsets(word, 'r'):
            ss_set = list(swn.senti_synsets(word, 'r'))
            if ss_set : 
                ss_set=ss_set[0]
        
        if ss_set:
            
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
    
    
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score,
                                         norm_pos_score, norm_neg_score,
                                         norm_final_score]],
                                         columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Objectivity',
                                                                       'Positive', 'Negative', 'Overall']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print (sentiment_frame)   
    return final_sentiment
            
      
## EVLUATES THE LEXICON-BASED SENTENCE SCORE WITH COMPARING TO ACTUAL TAGGED SENTENCE POLARITY(GIVEN IN THE DATASET)                                                         
def evaluate_lexicons(TRUE_LABELS,PREDICTED_LABELS) : 

    print ('Performance metrics:')
    display_evaluation_metrics(true_labels=TRUE_LABELS,
                               predicted_labels=PREDICTED_LABELS,
                               positive_class='positive')  
    print ('\nConfusion Matrix:'             )              
    display_confusion_matrix(true_labels=TRUE_LABELS,
                             predicted_labels=PREDICTED_LABELS,
                             classes=['positive', 'negative'])
    print ('\nClassification report:' )                        
    display_classification_report(true_labels=TRUE_LABELS,
                                  predicted_labels=PREDICTED_LABELS,
                                  classes=['positive', 'negative'])
    return


def main() : 
    
    print("*"*50)
    
    test_start=int(sys.argv[1])
    test_end=int(sys.argv[2])

    train_x,test_x,test_y=prepare_movie_dataset(0,1000,test_start,test_end)
    print("*"*50)    
    sentiwordnet_predictions = [analyze_sentiment_sentiwordnet_lexicon(review) for review in test_x]
    print("*"*50)
    evaluate_lexicons(test_y.tolist(),sentiwordnet_predictions)
    print("*"*50)
    print("Finished.")
    
if __name__=="__main__" : 
    main()