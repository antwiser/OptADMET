import _pickle as cPickle
import gzip

pattMap = cPickle.load(gzip.open('exp_sort.pkl.gz', 'rb'))
print(len(pattMap))