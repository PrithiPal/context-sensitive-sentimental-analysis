"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""
#import numpy as np 
#import pandas as pd 
import itertools 
import smtplib
import perc
import default
import sys
from collections import defaultdict
import gzip # use compressed data files
import copy, operator, optparse, sys, os
import re 
import time
import pprint

def perc_train(train_data, tagset, numepochs):
    t1=time.time()
    FEATURE_VEC_SENT=[] 
    FEATURE_DIC={}
    
    default_tag=tagset[0]
    
    #counter for numepochs
    e=0

    for epoch in range(numepochs) : 
        print("epoch {}".format(e))
        
        #counter for sentence index
        sent_ind=0
        ## ONE SENTENCE PARSING 
        for (sent_labels,sent_features) in train_data : 
            #returns a list of argmax tags that map to each word
            argmax_tag=perc.perc_test(FEATURE_DIC, sent_labels, sent_features, tagset, default_tag)       
            

            labels = copy.deepcopy(sent_labels)
            labels.insert(0, '_B-1 _B-1 _B-1')
            labels.insert(0, '_B-2 _B-2 _B-2') # first two 'words' are _B-2 _B-1
            labels.append('_B+1 _B+1 _B+1')
            labels.append('_B+2 _B+2 _B+2') # last two 'words' are _B+1 _B+2  
            #N is the number of words in a sentence
            N = len(labels)   

            ## for each word
            feat_index=0
            j=0
            for i in range(2, N-2):
                
                (feat_index, feats) = perc.feats_for_word(feat_index, sent_features)

                #["confidence",NN, B-NP]  
                fields = labels[i].split()    
                word = fields[0]
                actual_tag=fields[2]    
                
                #print("word_index = {}, word = {}, actual_tag = {}, argmax_tag= {}".format(i,word,actual_tag,argmax_tag[i]))
                
                amax_tag=argmax_tag[j]
                feat_list=[(x,actual_tag,amax_tag) for x in feats]
                
                '''
                    (f,b)
                    (feature, actual_tag)
                    (f,c)
                    (feature, argmax_tag)
                '''
                for (f,b,c) in feat_list :  
                    if b!=c : 
                        if (f,b) in FEATURE_DIC : 
                            FEATURE_DIC[(f,b)]=FEATURE_DIC[(f,b)]+1
                            #FEATURE_DIC[(f,b)]=FEATURE_DIC[(f,b)]+ (e +1)/2
                        else : 
                            #FEATURE_DIC[(f,b)]= (e +1)/2
                            FEATURE_DIC[(f,b)]=1
                            #FEATURE_DIC[(f,b)]= 0  
                    
                        if (f,c) in FEATURE_DIC : 
                            FEATURE_DIC[(f,c)]=FEATURE_DIC[(f,c)]-1
                            #FEATURE_DIC[(f,c)]=FEATURE_DIC[(f,c)] - (e +1)/2
                        else : 
                            FEATURE_DIC[(f,c)]=-1
                            #FEATURE_DIC[(f,c)]= -(e +1)/2
                            #FEATURE_DIC[(f,c)]= 0 

                #check next word with its argmax tag     
                j=j+1

            #check next sentence 
            sent_ind=sent_ind+1 
            '''
            #count key in FEATURE_DIC value = 0
            count_zero = 0
            count_one = 0
            count_neg = 0
            
            if(sent_ind == 1000):
                #pprint.pprint(local_keys)
                #print(len(local_keys))
                #pprint.pprint(FEATURE_DIC)
                break

            print(len(FEATURE_DIC))
                for key ,value in FEATURE_DIC.items():
                    if value == 0:
                        count_zero = count_zero +1
                    elif value == 1:
                        count_one = count_one +1
                    else:
                        count_neg = count_neg +1

                print(count_zero)
                print(count_one)
                print(count_neg)
                print(count_neg + count_one + count_zero)
            '''
            #print(FEATURE_DIC)
        #next iteration
        e=e+1
    
    print("Training finished in ",time.time()-t1)
    return FEATURE_DIC

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)

    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)

    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)


