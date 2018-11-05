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

import perc
import sys, optparse, os
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    # insert your code here
    # please limit the number of iterations of training to n iterations

    # >> INITIALIZE FEATURE VECTOR WITH WEIGHTS OF ZEROES

    # **NOTE: train_data HAS ONE ENTRY FOR EACH SENTENCE**

    print("len labeled_list: ", len(train_data))
    
    # go through each sentence
    # >> INDEX 'j' TP KEEP TRACK OF THE J-TH SENTENCE
    j=0

    words=[]
    labels=[]
    for (labeled_list, feat_list) in train_data:
        # **THE FIRST ITEM IN THE TUPLE IS THE LABELED LISTS**
        
        # # $$ WORDS FOR THE CURRENT SENTENCE $$
        # # >> x[1:n]
        # for itm in labeled_list:
        #     words.append(itm.split()[0])
        # print(words)
        # # $$$$

        # # $$ OUTPUT LABELS FOR THE CURRENT SENTENCE $$
        # # >> t[1:n]
        # for itm in labeled_list:
        #     labels.append(itm.split()[-1])
        # print(labels)
        # # $$$$

        # print(labeled_list)

        print("Len of feat_list: ", len(feat_list))

        # $$ GO THROUGH EACH FEATURE, STORE IN feat_vec $$
        count=0
        testbool=False
        print(feat_list[2])
        for ft in feat_list:
            # print(ft)
            # PUT FEATURE IF NOT IN THE feat_vec yet
            if ft not in feat_vec.keys():
                testbool = testbool or False
                feat_vec[ft] = 0
            else:
                testbool = testbool or True

            count+=1
        print(count)
        print(testbool)
        # $$$$

        # >> Use Viterbi algorthim (perc_test) here on the labeled sentence
        # output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, tagset[0])
        # print("\n".join(perc.conll_format(output, labeled_list)))
        # print()

        words=[]
        labels=[]
        # >> UPDATE 'j'
        j+=1


    # print("Printing feat_vec...")
    # print("len feat_vec: ", len(feat_vec))
    # for itm in feat_vec:
    #     print(itm)

    return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    # optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.dev"), help="input data, i.e. the x in \phi(x,y)")
    # optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.dev"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
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
    # print("len train_data: ", len(train_data))
    print("done.", file=sys.stderr)
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    # print(feat_vec)
    perc.perc_write_to_file(feat_vec, opts.modelfile)

