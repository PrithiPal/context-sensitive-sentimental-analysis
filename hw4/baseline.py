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


verbose=True



if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

if verbose:
    sys.stderr.write("Training with Dice's coefficient...\n")

bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), opts.num_sents)]
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
while k<2 : 
    if verbose:
        sys.stderr.write(' Iteration {} ..... \n'.format(k))
    k+=1
    for (n, (f, e)) in enumerate(bitext):
        for fi in f : 
            z=0
            for ej in e:
                z+=t[(fi,ej)] ## always going to be 1
            for ej in e:
                
                c=t[(fi,ej)]/z
                fe_count[(fi,ej)]+=c
                e_count[ej]+=c

    for (f,e) in fe_count : 
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
        bestj=0
        for j,ej in enumerate(e):
            if t[(fi,ej)]>bestp:
                bestp=t[(fi,ej)]
                bestj=j
      
        sys.stdout.write("%i-%i " % (f,bestj))
    sys.stdout.write("\n")

