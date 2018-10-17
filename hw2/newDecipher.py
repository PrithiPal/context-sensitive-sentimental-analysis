
# coding: utf-8

# # Homework: Decipherment

# In[1]:


from collections import defaultdict, Counter
import ngram
from ngram import *
import collections
import pprint
import math
import bz2
import numpy
import time
import pandas as pd
import numpy as np
pp = pprint.PrettyPrinter(width=45, compact=True)


# In[2]:


def read_file(filename):
    if filename[-4:] == ".bz2":
        with bz2.open(filename, 'rt') as f:
            content = f.read()
            f.close()
    else:
        with open(filename, 'r') as f:
            content = f.read()
            f.close()
    return content

def get_statistics(content, cipher=True):
    stats = {}
    content = list(content)
    split_content = [x for x in content if x != '\n' and x!=' ']
    length = len(split_content)
    symbols = set(split_content)
    uniq_sym = len(list(symbols))
    freq = collections.Counter(split_content)
    rel_freq = {}
    for sym, frequency in freq.items():
        rel_freq[sym] = (frequency/length)*100

    if cipher:
        stats = {'content':split_content, 'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
    else:
        stats = {'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
    return stats

def find_mappings(ciphertext, plaintext):
    mappings = defaultdict(dict)
    hypotheses = defaultdict(dict)

    for symbol in ciphertext['vocab']:
        for letter in plaintext['vocab']:
            hypotheses[symbol][letter] = abs(math.log((ciphertext['relative_freq'][symbol]/plaintext['relative_freq'][letter])))

    for sym in hypotheses.keys():
        winner = sorted(hypotheses[sym].items(), key=lambda kv: kv[1])
        mappings[sym] = winner[1][0]

    return mappings


# In[3]:


cipher = read_file("data/cipher.txt")
plaintxt = read_file("data/default.wiki.txt.bz2")


cipher_desc = get_statistics(cipher, cipher=True)
plaintxt_desc = get_statistics(plaintxt, cipher=False)

mapping = find_mappings(cipher_desc, plaintxt_desc)

english_text = []
for symbol in cipher_desc['content']:
    english_text.append(mapping[symbol])
decipherment = ('').join(english_text)
#print(decipherment)
print(cipher_desc['vocab'])


# In[4]:


symbol_list = [];
symbol_relFreq = [];
for x, y in cipher_desc["relative_freq"].items():
    symbol_list.append(x)
    symbol_relFreq.append(y)

index_names = {}
for i in range(54):
    index_names[i] = symbol_list[i]

test_data = numpy.ones((54,26))
df = pd.DataFrame(test_data, columns = plaintxt_desc['vocab'])
df=df.rename(index = index_names )


# ## LOADING NGRAM MODEL

# In[5]:


freq_dict=[ (k,v) for k,v in zip(cipher_desc['frequencies'].keys(),cipher_desc['frequencies'].values())]
sorted_freq_dict=sorted(freq_dict, key=lambda x:max([v[1] for v in freq_dict])-x[1])
sorted_symbols=[s[0] for s in sorted_freq_dict]

lm = ngram.LM("data/6-gram-wiki-char.lm.bz2", n=6, verbose=False)


# ## Maximum Context Score estimation

# In[6]:



## f = cipher symbol char,
def convert_to_bitstring(f,cipher_text) :
    return "".join(['o' if f == t else '.' for t in cipher_text])

## bs1,bs2 = bitstrings
def superimpose_bitstrings(bs1,bs2) :

    BS1=list(bs1)
    BS2=list(bs2)
    NEW_BS=[]
    for i in range(len(BS1)) :
        if BS2[i] == '.' and BS1[i] == '.' :
            NEW_BS.append('.')
        else :
            NEW_BS.append('o')


    return ''.join(NEW_BS)

## F is a list of cipher symbols
## cipher_text is string of cipher.
def convert_to_bitstring_multiple(F,cipher_text) :

    master_bs="."*len(cipher_text)
    for f in F :
        #print('f=',f)
        #print('cipher_text=',cipher_text)
        new_bitstring=convert_to_bitstring(f,cipher_text)
        #print('new_bitstring=',new_bitstring)
        master_bs = superimpose_bitstrings(master_bs,new_bitstring)
        #print("master_bs= ",master_bs)
    return master_bs

## cipher_text is all cipher in string
## f is single cipher char
## W is list of weights (6 lengths)
## phi is HS which is [(e,f) , (e,f) ...]
def calculate_maximum_context_score(cipher_text,f,W,phi) :
    new_phi = [i[1] for i in phi]
    new_phi.append(f)
    #print('new_phi=',new_phi)
    bitstring = convert_to_bitstring_multiple(new_phi,cipher_text)

    #print("bitstring= ",bitstring)
    contagious_o = re.findall(r'[o]+',bitstring)
    #print("contagiouso = ",contagious_o)

    contagious_lenghts = [len(i) for i in contagious_o]

    #print("contagious_lenghts = ",contagious_lenghts)
    N=6
    max_context=[float(len(list(filter(lambda x:x==i,contagious_lenghts)))) for i in range(1,N+1)]
    #print('max_context = ',max_context)
    term=np.multiply(W,max_context)
    #print('term= ',term)
    return sum(term)


def get_best_f_msscore(cipher_text,remaining_fs,weights,phi) :
    MS_SCORES=[]
    for f in remaining_fs :
        f_ms_score = calculate_maximum_context_score(cipher_text,f,weights,phi)
        MS_SCORES.append(f_ms_score)

    F_SCORES=dict([(i,j) for i,j in zip(MS_SCORES,remaining_fs) ])
    indmax=max(F_SCORES.keys())
    best_f=F_SCORES[indmax]
    #print(F_SCORES)
    #print(best_f)
    #print(indmax)
    return best_f

## EXAMPLE FOR TESTING

CIPHER='BURGER'
VOCAB='BUGER'
W=[1,1,1,1,2,3]
HS=[('', ''),('b','B'),('g','G')]
#calculate_maximum_context_score(CIPHER,'U',W,HS)
#get_best_f_msscore(CIPHER,['R','U'],W,HS)


# ## HELPER FUNCTIONS USED BY BEAM SEARCH

# In[7]:



## phi_obj is list of tuples.
def satisfy_ext_limits(phi_obj,nkeep) :

   # print(phi_obj)
    l = dict([(i[0],0) for i in phi_obj])
    for elem in phi_obj :
        l[str(elem[0])]+=1

    n_lengths=list(filter(lambda x:x>nkeep,list(l.values())))

    if n_lengths == [] :
        return True
    else :
        return False


## phi is [(e,f),(e,f) ... ]
## lm is language model from ngrams
## cipher is string of cipher
def score_partial_hypothesis(cipher, phi,lm) :

    ## reverse_phi is {f1:e1, f2:e2 ...}
    reverse_phi= dict([(i[1],i[0]) for i in phi ])
    #print('reverse_phi=',reverse_phi)

    f_phi_list = [i[1] for i in phi]
    #print("f_phi_list= ",f_phi_list)

    deciphered_tokens=[]
    arg1=[]

    overall_score=0
    for f in cipher :

        if f in f_phi_list :
            deciphered_tokens.append(reverse_phi[str(f)])
            arg1.append("o")
        else :
            deciphered_tokens.append(".")
            arg1.append(".")

    dt = "".join(deciphered_tokens)
    a1 = "".join(arg1)
    #print('a1 = ',a1)
    #print('dt tokens= ',dt)
    score=lm.score_bitstring(dt,a1)
    return score


def hist_prune(H,nkeep) :

    scores = [float(i[1]) for i in H]
    scores_s = sorted(H,key=lambda x:x[1])
    #print(scores_s)
    return scores_s[-nkeep:]


def explored_space(ch,nkeep) :
    phi=ch[0][0]
    if len(phi) > nkeep :
        return True
    else :
        return False


def get_sorted_syms(x1,x2) :
    freq_dict=[ (k,v) for k,v in zip(x2['frequencies'].keys(),x2['frequencies'].values())]
    sorted_freq_dict=sorted(freq_dict, key=lambda x:max([v[1] for v in freq_dict])-x[1])
    sorted_symbols=[s[0] for s in sorted_freq_dict]
    return sorted_symbols


def reached_threshold(thresh,hs) :
    scores=[i[1] for i in hs]
    thresh_scores=list(filter(lambda x:x>thresh,scores))
    if thresh_scores != [] :
        return True
    else :
        return False

def reached_depth(depth,hs) :
    phi=hs[0][0]
    d = len(phi)
    if d > depth :
        return True
    else :
        return False

def printifverbose(text, isverbose=False):
    if isverbose:
        print(text)


## ext_order is cipher tokens list
## ext_limits is number
## Vf is unique cipher tokens list
## Ve is unique english tokens list (a-z)
## nkeep is number
## cipher_text is string of all the cipher
## msscore_enabled is bool
def beam_search(ext_order, ext_limits,Vf,Ve,nkeep,cipher_text,thresh,max_depth,beam_width,msscore_enabled=False):
    # FOR 'BURGER' EXAMPLE:
    # ext_order: ['R', 'B', 'U', 'G', 'E']
    # Vf: ['U', 'E', 'B', 'G', 'R']
    # cipher_text: "burger"

    printifverbose(str("==startbeamsearch==").upper())

    Hs = []
    Ht = []
    # Hs and Ht will be of format '[phi, score]'
    # which is '[list, float]'
    # [([('', '')], 0)]
    cardinality = 0
    Hs.append(([('','')],0))
    #print(Hs[0])
    new_phi=[]

    W=[1,1,1,1,2,3]


    remaining_f=[]

  #  t1=time.time()
    for i in Vf :
        remaining_f.append(i)
  #  print("remaining_f copying time : ",time.time()-t1)

    current_hs=[]
    printifverbose("len(Vf): " + str(len(Vf)))


    while cardinality < len(Vf):  #line5
        print("depth reached =", cardinality)
        printifverbose("\t--beginwhile--")


        if msscore_enabled :
           # t1=time.time()
            f = get_best_f_msscore("".join(cipher_desc['content']),remaining_f,W,Hs) ## MAXIMUM CONTEXT IMPLEMENTATION
          #  print("get_best_f_msscore time : ",time.time()-t1)
        else :
            f = ext_order[cardinality]  # ORIGINAL IMPLEMENTATION


        printifverbose("\tCurrent cipher character (f): " + f + "\n")

        current_hs=[]
        for h in Hs:  #line7a

            printifverbose("\t\t--beginOuterloop--")
            phi=h[0]  #line7b
            printifverbose("\t\tlen(Ve):" + str(len(Ve)) + "\n")


            for e in Ve:  #line8
                printifverbose("\t\t\t--beginInnerloop--")

                printifverbose("\t\t\tcurrent (e) --> '" + e + "'")

                new_eandf=(e,f)  #line9a
                printifverbose("\t\t\tcurrent (e,f) --> ('" + e + "','" + f + "')")

                new_phi=[]

              #  t1=time.time()
                for p in phi :
                    new_phi.append(p)

                new_phi.append(new_eandf)
              #  print("new_phi copying time : ",time.time()-t1)


                printifverbose("\t\t\tϕ' = ϕ ∪ {(e,f)}")
                printifverbose("\t\t\t--> " + str(new_phi))

                # SCORE
               # t1=time.time()

                if satisfy_ext_limits(new_phi,ext_limits):  #line10
                    SCORE=score_partial_hypothesis(cipher_text,new_phi,lm)  #line11a
                    ht_entry=(new_phi,SCORE)  #line11b
                    Ht.append((ht_entry))  #line 11c

               # print("ext_limits finding scores time : ",time.time()-t1)

                printifverbose("\t\t\t(ϕ', SCORE(ϕ'))")
                printifverbose("\t\t\t--> " + str(ht_entry) + "   ##Add to Ht")
                printifverbose("\t\t\t--endInnerloop--\n")

            ## INNER LOOP ENDS

            printifverbose("\t\tHt --> " + str(Ht)) # + "\n")
            printifverbose("\t\t--endOuterloop--\n")

           # t1=time.time()
            prune = hist_prune(Ht,nkeep)
          #  print("hist_prune time : ",time.time()-t1)

            for p in prune :
                current_hs.append(p)

            Ht=[]

        cardinality = cardinality + 1  #line13

        printifverbose("\tHt after prunning --> " + str(Ht)) # + "\n")
        printifverbose("\n\tHs = Ht\n\tHs --> " + str(Ht)) # + "\n")

        Hs = current_hs

        SATISFIED=reached_threshold(thresh,Hs)
        DEPTH_REACHED=reached_depth(max_depth,Hs)
        if SATISFIED and DEPTH_REACHED :
            print('depth and satisfied exiting..')
            break

        if len(Hs) >= beam_width :
            break

        Ht=[]  #line15

        printifverbose("\t--endwhile--" + "\n")
        remaining_f.remove(f)


    printifverbose("==endbeamsearch==" + "\n")
    return Hs  #WINNER(Hs)


# ## TOY EXAMPLE TO TEST BEAM SEARCH AND EVALUATE RESULTS

# In[54]:


## TESTING BEAM SEARCH ON SIMPLE 1:1 SUBSITUTION CIPHER
def score_beam(hs,real_phi) :
    set1=set(hs)
    set2=set(real_phi)
    common=set1.intersection(set2)
    return len(common)/len(set1)

def main() :


    def score_beam(hs,real_phi) :
        set1=set(hs)
        set2=set(real_phi)
        common=set1.intersection(set2)
        return len(common)/len(set1)


    import string
    sample_text="onceuponatimetherewasamannamedekamwhodiedbecausehegotstruckbylightning"
    #sample_text="thisismine"
    sample_text_list= list(sample_text)
    ILLEGAL=string.punctuation
    ILLEGAL=ILLEGAL+"\ \’\‘"

    sample_text="".join([i for i in sample_text_list if i not in ILLEGAL])
    #print("sample_text=",sample_text)

    cipher_text = sample_text.upper() ## BURGER is the CIPHER

    ## x2 is plaintext  get_statistics() object
    ## x1 is ciphertext get_statistics() object

    def get_sorted_syms(x1,x2) :
        freq_dict=[ (k,v) for k,v in zip(x2['frequencies'].keys(),x2['frequencies'].values())]
        sorted_freq_dict=sorted(freq_dict, key=lambda x:max([v[1] for v in freq_dict])-x[1])
        sorted_symbols=[s[0] for s in sorted_freq_dict]
        return sorted_symbols

    cipher_desc = get_statistics(cipher_text,cipher=True)
    ss=get_sorted_syms(plaintxt_desc,cipher_desc)
    W=[1.0,1.0,1.0,1.0,2,3]

    ALPHA1="abcdefghijklmnopqrstuvwxyz"
    ALPHA2="ABCDEFGHIJKLMNOPQRSTUVWXYZ" ## CIPHER
    REAL_PHI=[(a,b) for a,b in zip(ALPHA1,ALPHA2)]

    EXT_ORDER=ss
    EXT_LIMIT=1
    KEEPS=1
    VF=cipher_desc['vocab'] ## LIST OF CIPHER TOKENS
    VE=plaintxt_desc['vocab']
    CUTOFF_THRESH=-1000
    MAX_DEPTH=len(VF)
    BEAM_WIDTH=10000000
    print("--- INPUT PARAMS -----")

    print("EXT_ORDER= ",EXT_ORDER)
    print("EXT_LIMIT= ",EXT_LIMIT)
    print("VF= ",VF)
    print("VE= ",VE)
    print("KEEPS = ",KEEPS)
    print("cipher_text = ",cipher_text)
    print("CUTOFF_THRESH = ",CUTOFF_THRESH)
    print("MAX_DEPTH = ",MAX_DEPTH)
    print("BEAM_WIDTH = ",BEAM_WIDTH)


    print("------ BEAM SEARCH RUNNING -----")

    final_hs=beam_search(EXT_ORDER,EXT_LIMIT,VF,VE,KEEPS,cipher_text,CUTOFF_THRESH,MAX_DEPTH,BEAM_WIDTH,msscore_enabled=False)
    print("final_hs = ",final_hs)
    print("------ BEAM SEARCH FINISHED -----")

    ## EVALUATE THE EFFECTIVENESS OF ALGORITHM. COMPARES THE BEST HYPOTHESIS GENERATED BY ALGORITHM
    ##AGAINST THE ACTUAL CORRECT CIPHERS (COMPARED AGAINST REAL CIPHER KEY)

    print("------ EVALUATING RESULTS-------")

    phi=[f[0] for f in final_hs]
    REAL_SCORES=[]
    for p in phi :
        REAL_SCORES.append(score_beam(p,REAL_PHI))

    ind1=np.argmax([f[1] for f in final_hs] )
    ind2=np.argmax(REAL_SCORES)

    print("BEST RULES ACCORDING TO NGRAM SCORING [SCORE = {}]\n{} ".format(score_beam(final_hs[ind1],real_key),final_hs[ind1],))
    print("BEST ACTUAL SCORE (COMPARING WITH REAL CIPHER KEY) [SCORE={}]\n{} ".format(score_beam(final_hs[ind2],real_key),final_hs[ind2]))


def main2() :

    def score_beam(hs,real_phi) :
        set1=set(hs)
        set2=set(real_phi)
        common=set1.intersection(set2)
        return len(common)/len(set1)

    W=[1.0,1.0,1.0,1.0,2,3]
    cipher_desc = get_statistics(cipher,cipher=True)
    ss=get_sorted_syms(plaintxt_desc,cipher_desc)

    cipher_text="".join(cipher_desc['content'])
    EXT_ORDER=ss
    EXT_LIMIT=3
    KEEPS=2
    VF=cipher_desc['vocab'] ## LIST OF CIPHER TOKENS
    VE=plaintxt_desc['vocab']
    CUTOFF_THRESH=-1000
    MAX_DEPTH=12
    BEAM_WIDTH=10000


    print("--- INPUT PARAMS -----")


    print("EXT_ORDER= ",EXT_ORDER)
    print("EXT_LIMIT= ",EXT_LIMIT)
    print("VF= ",VF)
    print("VE= ",VE)
    print("KEEPS = ",KEEPS)
    print("CUTOFF_THRESH = ",CUTOFF_THRESH)
    print("MAX_DEPTH = ",MAX_DEPTH)
    print("BEAM_WIDTH = ",BEAM_WIDTH)
    print("cipher_text = ",cipher_text)


    print("------ BEAM SEARCH RUNNING -----")
    final_hs=beam_search(EXT_ORDER,EXT_LIMIT,VF,VE,KEEPS,cipher_text,CUTOFF_THRESH,MAX_DEPTH,BEAM_WIDTH,msscore_enabled=False)
    print("final_hs = ",final_hs)
    print("------ BEAM SEARCH FINISHED -----")



    ## EVALUATE THE EFFECTIVENESS OF ALGORITHM. COMPARES THE BEST HYPOTHESIS GENERATED BY ALGORITHM
    ##AGAINST THE ACTUAL CORRECT CIPHERS (COMPARED AGAINST REAL CIPHER KEY)


    print("------ EVALUATING RESULTS-------")

    keyfile = read_file("data/key.txt")
    key_desc = get_statistics(keyfile)
    real_key = list(zip(cipher_desc['content'], key_desc['content'])) ## [(e,f) ...]

    phi=[f[0] for f in final_hs]
    REAL_SCORES=[]
    for p in phi :

        REAL_SCORES.append(score_beam(p,real_key))


    ind1=np.argmax([f[1] for f in final_hs] )
    ind2=np.argmax(REAL_SCORES)

    print("BEST RULES ACCORDING TO NGRAM SCORING [SCORE = {}]\n{} ".format(score_beam(final_hs[ind1][0],real_key),final_hs[ind1]))
    print("BEST ACTUAL SCORE (COMPARING WITH REAL CIPHER KEY) [SCORE={}]\n{} ".format(score_beam(final_hs[ind2][0],real_key),final_hs[ind2]))
    print("REAL KEY : ",real_key)

if __name__ == '__main__' :
    main2()

# ## FOR TOY EXAMPLE, FINDING ACCURACY OF ALGORITHM (NGRAM SCORE VS CIPHER-KEY SCORE)
# In[51]:


#ind=np.argmin([f[1] for f in final_hs] )
#final_hs




## here same for both. This may mean that our score function is working correctly.
