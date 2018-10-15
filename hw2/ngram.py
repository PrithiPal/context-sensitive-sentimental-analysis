import sys, bz2, re, string
from collections import namedtuple

# # A language model scores sequences, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = LM(filename, n=6, verbose=False)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for t in sentence:
#   (lm_state, logprob) = lm.score(lm_state, t)
#   logprob += logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
class LM:
    def __init__(self, filename, n=6, verbose=False):
        print("Reading language model from {}...".format(filename), file=sys.stderr)
        self.table = {}
        self.n = n
        self.history = n-1
        self.verbose = verbose
        for line in bz2.open(filename, 'rt'):
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))
                self.table[ngram] = ngram_stats(logprob, backoff)
        print("Done.", file=sys.stderr)

    def begin(self):
        return ("<s>",)

    def score(self, state, token):
        #print("[SCORE] : .. SCORE .. : {} + {} .....".format(state,token))
        ngram = state + (token,)
        score = 0.0
        while len(ngram)> 0:
            if ngram in self.table:
              #  print("[SCORE] : NGRAM FOUND IN FILE : ",ngram)
                return (ngram[-self.history:], score + self.table[ngram].logprob)
            else: #backoff
                
                score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
                ngram = ngram[1:]
              #  print("[SCORE] : NGRAM BACKOFF : ",ngram)
       # print("[SCORE] : {} + {} score = {}".format(state,token,score))
        return ((), score - 99.0) # bad score for missing unigrams
    
    def end(self, state):
        return self.score(state, "</s>")[1]

    def clean_seq(self, sequence):
        return(sequence.translate(dict.fromkeys(map(ord, string.punctuation + ' '), None)).lower())

    def maybe_write(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr)

    def score_seq(self, sequence):
        
        lm_state = self.begin()
        lm_logprob = 0.0 
        
       # print("[SCORE_SEQ] : --- SCORE SEQ : {}--------".format(sequence))
        for token in list(self.clean_seq(sequence)):
            
            self.maybe_write("state: {}".format(lm_state + (token,)))
           # print("[SCORE_SEQ] : new state: {}".format(lm_state + (token,)))
            
            
            ## READ THE LOGPROBABILITYH FOR THE NGRAM FROM FILE.
            (lm_state, logprob) = self.score(lm_state, token)
            
            
            ## HERE IS THE LOGIC TO COMPUTE ACTUAL SCORE. 
            
            ## ORIGINAL HEURISTIC : SUM OVER
            lm_logprob += logprob
            self.maybe_write("logprob={}".format(logprob))
            
           # print("[SCORE_SEQ] : state : {}, token : {}, state_score : {}, LM SCORE : {}".format(lm_state,token,logprob,lm_logprob))
        #
        lm_logprob += self.end(lm_state)
        return lm_logprob

    def get_bitstring_spans(self, bitstring):
        """get a list of spans that are contiguous and have 'o' in
        the string position. ignore '.' positions"""
       # print("[GET_BITSTRING_SPANS] : BIT STRING = ",{ i.span()[0] : i.span()[1] for i in re.finditer('o', bitstring) })
        return { i.span()[0] : i.span()[1] for i in re.finditer('o', bitstring) }

    def score_bitstring(self, sequence, bitstring):
     
        spans = self.get_bitstring_spans(bitstring)
      #  print("[SCORE_BITSTRING] : spans :  ",spans)


        seq_by_bits = [ sequence[i] if i in spans else '\t' for i in range(len(sequence)) ]
      #  print("[GET_BITSTRING_SPANS] seq_by_bite  : ,",seq_by_bits)
        self.maybe_write("seq_by_bits: {}".format(seq_by_bits))
        #print("seq_by_bits: {}".format(seq_by_bits))
        
        lm_state = self.begin()
        lm_logprob = 0.0 
        for token in list(seq_by_bits):
            if token == '\t': # should we skip this token?
                lm_state = ()
                continue
            self.maybe_write("state: {}".format(lm_state + (token,)))
            
            
            (lm_state, logprob) = self.score(lm_state, token)
            lm_logprob += logprob
            self.maybe_write("logprob={}".format(logprob))

            
        lm_logprob += self.end(lm_state)
        return lm_logprob

if __name__ == '__main__':
    sequence = 'In a few cases, a multilingual artifact has been necessary to facilitate decipherment, the Rosetta Stone being the classic example. Statistical techniques provide another pathway to decipherment, as does the analysis of modern languages derived from ancient languages in which undeciphered texts are written. Archaeological and historical information is helpful in verifying hypothesized decipherments.'

    lm = LM("data/6-gram-wiki-char.lm.bz2", n=6, verbose=False)
    
    #lm_logprob = lm.score_seq(sequence)
    #print("TOTAL LM LOGPROB for \"{}\": {}".format(sequence, lm_logprob), file=sys.stderr)

    #s1 = 'zkxxuqxzpuq'
   # s2 = 'thisisatest'
    
    ## HERE THE NGRAM ESTIMATOR ACTING LIKE TRIGRAM WHY??.
    #print("TOTAL LM LOGPROB for \"{}\": {}".format(s1, lm.score_seq(s1)), file=sys.stderr)
   # print("TOTAL LM LOGPROB for \"{}\": {}".format(s2, lm.score_seq(s2)), file=sys.stderr)

   # print("--------- BITSTRING ---------------")
   # print(lm.get_bitstring_spans('..oo...ooo..')) 
    print(lm.score_bitstring('thisisatest', 'ooooooooooo'))
