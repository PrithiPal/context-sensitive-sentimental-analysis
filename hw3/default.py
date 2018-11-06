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

import numpy as np 
import pandas as pd 
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

def perc_train(train_data, tagset, numepochs):
    t1=time.time()
    FEATURE_VEC_SENT=[] 
    FEATURE_DIC={}
    GLOBAL_DIC=[]
    
    default_tag=tagset[0]
    
    e=0
    for epoch in range(numepochs) : 
        print("epoch {}".format(e))
        
        sent_ind=0
        
        ## ONE SENTENCE PARSING 
        for (sent_labels,sent_features) in train_data : 
            #print('sent ',sent_ind)
            
            FEATURE_DIC={}
            if sent_ind==0 : 
                GLOBAL_DIC=FEATURE_DIC

            
            argmax_tag=perc.perc_test(GLOBAL_DIC, sent_labels, sent_features, tagset, default_tag)           
            
            
            
            ## for each word
            for w,i in zip(sent_labels,range(len(sent_labels))) : 
                
                
                
                (feat_index, feats) = perc.feats_for_word(i, sent_features)    
                actual_tag=re.findall(r"([A-Z\-]*)$",w)[0]
                word=re.findall(r"^([A-Za-z]*) *",w)[0]
                #print("word_index = {}, word = {}, actual_tag = {}, argmax_tag= {}".format(i,word,actual_tag,argmax_tag[i]))
               
                ## for each (word_feature,true_tag)
                local_keys=FEATURE_DIC.keys()
                global_keys=GLOBAL_DIC.keys()
                
                for (f,t) in itertools.product(feats,[actual_tag]) : 
                    #print("({},{}) {}".format(f,t,argmax_tag[i]))
                    if t!= argmax_tag[i] : 
                        #print("yes")  
                        if (f,t) in local_keys : 
                            FEATURE_DIC[(f,t)]=FEATURE_DIC[f,t]+1
                        else : 
                            FEATURE_DIC[(f,t)]=1  
                        if (f,argmax_tag[i]) in local_keys : 
                            FEATURE_DIC[(f,argmax_tag[i])]=FEATURE_DIC[(f,argmax_tag[i])]-1
                        else : 
                            FEATURE_DIC[(f,argmax_tag[i])]=-1
                
            #print(list(GLOBAL_DIC.values())[:10])
                            
            ## appending to global feature vector
            
            local_keys=FEATURE_DIC.keys()
            global_keys=GLOBAL_DIC.keys()
            
            t_start=time.time()
            if sent_ind!=0 : 
                ## appending new features of current feature vector if not already in global
                for k in list(local_keys) : 
                    global_k=list(global_keys)
                    if k not in global_k : 
                        if FEATURE_DIC[k]!=0 : 
                            GLOBAL_DIC[k]=FEATURE_DIC[k]
                    else : 
                        if GLOBAL_DIC[k]+FEATURE_DIC[k]!=0 : 
                            GLOBAL_DIC[k]=GLOBAL_DIC[k] + FEATURE_DIC[k]
            
            print("sentence {} trained in {}".format(sent_ind,time.time()-t_start))
            sent_ind=sent_ind+1
            
            
        e=e+1

        
    print("Training finished in ",time.time()-t1)
    return GLOBAL_DIC
                

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

