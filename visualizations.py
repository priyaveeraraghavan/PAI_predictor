import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg') ## necessary to force a different backend
import matplotlib.pyplot as plt
import sys


def get_precision_recall(classifier_fname):
    classifier_output = np.loadtxt(classifier_fname)
    y_true = [int(x) for x in classifier_output[:,1]]
    y_scores = classifier_output[:,4]
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
    return precision, recall, thresholds

def get_roc(y_true, y_scores):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    print fpr, tpr
    """thresholds = sorted(list(set(y_scores)))
    
    # true positive rate, or sensitivity
    def tpr_fpr(thresh):
        y_pred = [1 if x > thresh else 0 for x in y_scores] 
        tp = sum([x*y for x, y in zip(y_true, y_pred)])
        fp = sum(y_pred) - tp

        return np.array([tp/float(len(y_true)), fp/float(len(y_true))])

    tpr_fpr_list = [tpr_fpr(x) for x in thresholds]
    print tpr_fpr_list
    sensitivity = tpr_fpr_list[:,0].tolist()
    false_positive = tpr_fpr_list[:,1].tolist()"""

    return fpr, tpr, thresholds
    
        
def load_file(classifier_fname):
    classifier_output = np.loadtxt(classifier_fname)
    y_true = [int(x) for x in classifier_output[:,0]]
    y_scores = classifier_output[:,1]
    return y_true, y_scores

def main():
    outfile = sys.argv[1]
    lines = []
    plt.figure(figsize=(7.5, 10))
    for fname in sys.argv[2:]:
        y_true, y_scores = load_file(fname)
        false_positive_rate, true_positive_rate, threshold = get_roc(y_true, y_scores)
        line = plt.plot(false_positive_rate, true_positive_rate, label=fname)
        lines.append(line)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate (1-specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
    plt.savefig(outfile, bbox_layout='tight')

main()
