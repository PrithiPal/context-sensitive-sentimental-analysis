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
    default_tag=tagset[0]
    WEIGHT_VECTOR=[]

    history_list =[]
    e=0
    for epoch in range(numepochs) : 
        print("epoch {}".format(e))
        MISTAKES=0
        sent_ind=0
        ## ONE SENTENCE PARSING 
        epoch_start=time.time()
        
        for (sent_labels,sent_features) in train_data : 
            
            #print("sentence=",sent_labels)
            #argmax_tag=[]
             
            if epoch==0 : 
                FEATURE_DIC={}
            elif epoch>0: 
                FEATURE_DIC=WEIGHT_VECTOR[sent_ind]
            
          
            
            argmax_tag=perc.perc_test(FEATURE_DIC, sent_labels, sent_features, tagset, default_tag)   
            
                
            labels = copy.deepcopy(sent_labels)
            labels.insert(0, '_B-1 _B-1 _B-1')
            labels.insert(0, '_B-2 _B-2 _B-2') # first two 'words' are _B-2 _B-1
            labels.append('_B+1 _B+1 _B+1')
            labels.append('_B+2 _B+2 _B+2') # last two 'words' are _B+1 _B+2 
            #N is the number of words in a sentence 
            N = len(labels)   
            
            SUM=0
            ## for each word chunk
            feat_index=0
            j=0
            for i in range(2, N-2):
                
                (feat_index, feats) = perc.feats_for_word(feat_index, sent_features)  
               # print(feats)
                fields = labels[i].split()    
                
                
                word = fields[0]
                actual_tag=fields[2]    
               
        
                #print("word_index = {}, word = {}, actual_tag = {}, argmax_tag= {}".format(i,word,actual_tag,argmax_tag[i])
                amax_tag=argmax_tag[j]
                
                local_keys=list(FEATURE_DIC.keys())
                feat_list=[(x,actual_tag,amax_tag) for x in feats]

                
                if actual_tag!= amax_tag : 
                    MISTAKES+=1
                
                ## UPDATE THE WEIGHTS HERE.
                
                '''
                    (f,b)
                    (feature, actual_tag)
                    (f,c)
                    (feature, argmax_tag)
                '''
                for (f,b,c) in feat_list :  
                    
                    if b!=c : 
                        SUM+=1
                        if (f,b) in local_keys : 
                            FEATURE_DIC[f,b]+=1
                        else :
                            FEATURE_DIC[f,b]=1
                    
                        if (f,c) in local_keys : 
                            FEATURE_DIC[f,c]-=1
                        else : 
                            FEATURE_DIC[f,c]=-1
                        
                j=j+1                 
            
            
           
            
            if epoch == 0 : 
                WEIGHT_VECTOR.append(dict(list(FEATURE_DIC.items())))   
            elif epoch>0 :
                OLDER_FEATURE_DIC=WEIGHT_VECTOR[sent_ind]
                CURRENT_FEATURE_DIC=FEATURE_DIC
                SUM_FEATURE_DIC={}

                
                for a in OLDER_FEATURE_DIC.keys() : 
                 
                    SUM_FEATURE_DIC[a]=OLDER_FEATURE_DIC[a]+CURRENT_FEATURE_DIC[a]
                    
                
            
                WEIGHT_VECTOR[sent_ind]=SUM_FEATURE_DIC.copy()
                
                
            sent_ind=sent_ind+1  
            
        print('Num of Mistakes=',MISTAKES)   
        print('epoch {} ended in {}'.format(e,time.time()-epoch_start))
        e=e+1
        
        
    print("Training finished in ",time.time()-t1)
    return WEIGHT_VECTOR
    #return FEATURE_DIC            
            



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
    print(type(feat_vec))
    #F={} 
    #TESTING
    F= defaultdict(int)
    
    for i in range(len(feat_vec)) : 
        a=feat_vec[i].copy()
        F.update(a)
    
    print(type(F))
    feat_vec2=F.copy()
    print(type(feat_vec2))
    '''
    cnt_item = 0
    for key, value in feat_vec2.items() :
        print (key, value)
        cnt_item = cnt_item +1

        if cnt_item == 30:
            break
    '''
    perc.perc_write_to_file(feat_vec2, opts.modelfile)

