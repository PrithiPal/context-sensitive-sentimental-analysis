#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
from itertools import islice

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
e_with_null_data= "%s.%s" % (os.path.join(opts.datadir, '{}_null'.format(opts.fileprefix)), opts.english)


verbose=True



if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

if verbose:
    sys.stderr.write("Training with Dice's coefficient...\n")
## Bitext is the list of lists(two lists inside. first is french and other list is english)

bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), opts.num_sents)]


#sys.stderr.write(str(bitext) + '\n')
e_count = defaultdict(int)
f_count = defaultdict(int)
fe_count = defaultdict(int)
t = defaultdict(int)



if verbose:
    sys.stderr.write('Updating the fe_count and e_count dicts\n')


## POPULATING f_count, e_count and fe_count
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")


## INITIALIZING THE t
if verbose:
    sys.stderr.write('initializing t\n')
for (n, (f, e)) in enumerate(bitext):
    for fi in f : 
        for ej in e : 
            #sys.stderr.write('{} -> {} \n'.format(fi,ej))
            t[(fi,ej)]=1/len(f)
            
        
        
if verbose: 
    sys.stderr.write('Training baseline model\n')             


k=0
## POPULATING THE T 

## PARAM FOR THE WORD-ALIGNMENT MODEL.
smoothing=True
NUM_ITERATIONS=1

## If smoothing
if smoothing:
    sys.stderr.write('With Smoothing\n')
V=10000 ## added vocab size; value borrowed from the research paper. 
n=0.01 ## added count values borrowed from the research paper. 

# Null value
nullVal="<null>"

while k<NUM_ITERATIONS : 
    if verbose:
        sys.stderr.write(' Iteration {} ..... \n'.format(k))
    k+=1
    for (n, (f, e)) in enumerate(bitext):
        for fi in f : 
            z=0
            # Include Null before anything else
            z=t[(fi,nullVal)]
            for ej in e:
                
                z+=t[(fi,ej)] ## always going to be 1
                
            # Include Null before computing anything else
            c=t[(fi,nullVal)]/z
            fe_count[(fi,nullVal)]+=c
            e_count[nullVal]+=c
            for ej in e:
                # if ej=='<null>' : 
                #     c=t[(fi,ej)]/z
                #     fe_count[(fi,ej)]*=c
                #     e_count[ej]*=c
                # else :
                c=t[(fi,ej)]/z
                fe_count[(fi,ej)]+=c
                e_count[ej]+=c

    for (f,e) in fe_count : 
        if smoothing:
            ## smoothened observed counts
            t[(f,e)]=( fe_count[(f,e)]+n )/( e_count[e]+(n*V) ) 
        else:
            ## Default baseline counts 
            t[(f,e)]=fe_count[(f,e)]/e_count[e]
      

#if verbose:
    #sys.stderr.write(str(t))
    #sys.stderr.write(str(e_count))
        
if verbose:
    sys.stderr.write('Training Finished.\n')
    sys.stderr.write('Finding best alignments\n')



    

## FINDING THE BEST ALIGNMENT
for (f, e) in bitext:
    for f,fi in enumerate(f)  :
        bestp=0
        # Make Null the default bestp
        bestp=t[(fi,nullVal)]
        bestj=0
        for j,ej in enumerate(e):
            if t[(fi,ej)]>bestp:
                bestp=t[(fi,ej)]
                bestj=j
        # ONLY OUTPUT IF bestj IS NOT 0 
        # - i.e when bestj wasn't changed,
        #       meaning t[(fi,ej)] wasn't better than t[(fi,nullVal)])
        if bestj!=0:
            sys.stdout.write("%i-%i " % (f,bestj))
        #     sys.stderr.write('NULL ')
        #     break
        # else:
        #     sys.stdout.write("%i-%i " % (f,bestj))
    sys.stdout.write("\n")

