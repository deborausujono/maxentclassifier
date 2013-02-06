#!/sw/bin/python

import math
import sys
import glob
import pickle
import optimize
import Numeric as Num
from dicts import DefaultDict

# In the documentation and variable names below "class" is the same
# as "category"

constraints = DefaultDict(0)

def train_maxent (dirs):
    """Train and return a MaxEnt classifier.  
    The datastructure returned is dictionary whose keys are
    ('classname','word') tuples.  The values in the dictionary are
    the parameters (lambda weights) of the classifier.
    Note that this method does not return the list of classnames, 
    but the caller has those available already, since it is exactly the
    'dirs' argument.  

    If you need to recover the classnames from the diciontary itself, 
    you'd need to do something like:
    maxent = train_maxent(dirs)
    classes = list(set([c for (c,v) in maxent.keys()]))

    Some typical usage:
    dirs = ['spam','ham'] # where these are sub-directories of the CWD
    maxent = train_maxent(dirs)
    # interested in seeing the weight of "nigerian" in the "spam" class?
    lambda_spam_nigerian = maxent[('spam','nigerian')]
    # to classify a document
    scores = classify(maxent,dirs,"spam/file123")
    """
    classes = dirs
    maxent = DefaultDict(0)
    # Gather the "constraints" and initialize all-zero maxent dictionary
    global constraints
    for cls in classes:
	maxent[(cls,'DEFAULT')] = 0
	print cls
	for file in glob.glob(cls+"/*"):
	    for word in open(file).read().split():
		word = word.lower()
		constraints[(cls,word)] += 1
		for clss in classes:
		    maxent[(clss,word)] = 0
    # Remember the maxent features, and get the starting point for optimization
    features = maxent.keys()
    lambda0 = maxent.values()
    
    # Here call an optimizer to find the best lambdas
    lambdaopt = optimize.fminNCG(value, lambda0, gradient, args=(features,dirs), printmessg=1, maxiter=5)
    
    # Put the final optimal parameters are in returned dictionary
    assert maxent.keys() == features # Make sure the keys have not changed order
    maxent2 = dict([(k,v) for (k,v) in zip(maxent.keys(),lambdaopt)])
    return maxent2

def compute_expectations(lambdas, keys, dirs):
    """
    pnum = {}
    pden = {}
    for (cls, word) in keys:
        pnum[(cls, word)] = math.exp(lambdas[keys.index((cls, word))])
        if word in pden:
            pden[word] += pnum[(cls, word)]
        else:
            pden[word] = pnum[(cls, word)]
    """
            
    expectations = {}
    for dir in dirs:
	for file in glob.glob(dir+"/*"):
            #Calculate P(c'|d) for each c' in c
            words = open(file).read().split()
            words = [word.lower() for word in words]
            
            sum = {}
            for word in words:
                for cls in dirs:
                    if cls in sum:
                        sum[cls] += lambdas[keys.index((cls, word))]
                    else:
                        sum[cls] = lambdas[keys.index((cls, word))]

            pden = 0
            for cls in dirs:
                pden += math.exp(sum[cls])

            p = {}
            for cls in dirs:
                p[cls] = math.exp(sum[cls])/pden

            #Fill in expectations dictionary
            for word in words:
                for cls in dirs:
                    if (cls, word) in expectations:
                        expectations[(cls, word)] += p[cls]
                    else:
                        expectations[(cls, word)] = p[cls]

    #Calculate P(cls|'DEFAULT')
    word = 'DEFAULT'
    for cls in dirs:
        """
        p = 1.*pnum[(cls, word)]/pden[word]
        if (cls, word) in expectations:
            expectations[(cls, word)] += p
        else:
            expectations[(cls, word)] = p
        """
        expectations[(cls, word)] = 0
                    
    return expectations
                    

def gradient (lambdas, keys, dirs):
    expectations = compute_expectations(lambdas, keys, dirs)

    g = []
    for f in keys:
        g.append(expectations[f]-constraints[f]+(lambdas[keys.index(f)]/.5))
        #g.append(constraints[f]-expectations[f]-(lambdas[keys.index(f)]/.5))

    return Num.asarray(g)

def value (lambdas, keys, dirs):
    """Return the log-likelihood of the true labels 
    of the documents in the directories in the list 'dirs',
    using the parameters given in lambdas, where those lambdas
    correspond to the (word,class) keys given in 'keys'."""
    print dirs

    # Build a MaxEnt classifier dictionary from the keys and lambdas
    maxent = dict([(k,v) for (k,v) in zip(keys,lambdas)])
    # Use this MaxEnt classifier to classify all the documents in dirs
    # Accumulate the log-likelihood of the correct class
    classes = dirs
    total_log_prob = 0
    for c in classes:
	for file in glob.glob(c+"/*"):
	    probs = classify(maxent, classes, file)
	    # Pull out of 'probs' the log-prob of class c
	    # Remember, probs looks like [(0.85,'spam'), (0.15,'ham')]
	    true_class = [x[0] for x in probs if x[1] == c]
	    true_class_prob = true_class[0]
	    total_log_prob += math.log(true_class_prob)

    gaussian = 0
    for l in lambdas:
        gaussian -= (l*l)

    # Return the NEGATIVE total_log_prob because fminNCG minimizes, 
    # and we want to MAXIMIZE log probability
    # TO DO: Incorporate a Gaussian prior on parameters here!
    return - (total_log_prob+gaussian)

def classify (maxent, classes, filename):
    """Given a trained MaxEnt classifier returned by train_maxent(), and
    the filename of a test document, d, return an array of tuples, each
    containing a class label; the array is sorted by log-probability 
    of the class, log p(c|d)"""
    scores = []
    #print 'Classifying', filename
    for c in classes:
	# Put in the weight for the default feature
	score = maxent[(c,'DEFAULT')]
	# Put in the weight for all the words in the document
	for word in open(filename).read().split():
	    word = word.lower()
	    if (c, word) in maxent: score += maxent[(c,word)]
	scores.append(score)
    # exp() and normalize the scores to turn them into probabilities
    minimum = min(scores)
    print scores
    scores = [math.exp(x-minimum) for x in scores]
    normalizer = sum(scores)
    scores = [x/normalizer for x in scores]
    # make the scores list actually contain tuples like (0.84,"spam")
    scores = zip(scores,classes)
    scores.sort()
    return scores


if __name__ == '__main__':
    print 'argv', sys.argv
    print "Usage:", sys.argv[0], "classdir1 classdir2 [classdir3...] testfile"
    dirs = sys.argv[1:-1]
    testfile = sys.argv[-1]
    maxent = train_maxent (dirs)
    print classify(maxent, dirs, testfile)
    pickle.dump(maxent, open("maxent.pickle",'w'))

# E.g. type at command line
# python maxent.py spam ham spam/file123
# You will need the Numeric and MLab libraries to be installed.
# Otherwise you can implement your own conjugate gradient method, 
# which isn't very hard either.  For example, see "Numeric Recipes in C".
