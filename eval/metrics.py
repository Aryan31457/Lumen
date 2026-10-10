import numpy as np
from sklearn.metrics import adjusted_rand_score

def pairwise_precision_recall(true_labels,pred_labels):
    tp=fp=fn=0
    for i in range(len(true_labels)):
        for j in range(i+1,len(true_labels)):
            same_true=true_labels[i]==true_labels[j]; same_pred=pred_labels[i]==pred_labels[j]
            tp+=same_true and same_pred; fp+=(not same_true) and same_pred; fn+=same_true and not same_pred
    return tp/max(1,tp+fp),tp/max(1,tp+fn)
def ari(true_labels,pred_labels): return adjusted_rand_score(true_labels,pred_labels)
def recall_at_k(ranked,expected,k): return len(set(ranked[:k])&set(expected))/max(1,len(expected))
def mrr(ranked,expected): return next((1/i for i,item in enumerate(ranked,1) if item in expected),0.0)
def ndcg_at_k(ranked,expected,k):
    dcg=sum(1/np.log2(i+2) for i,item in enumerate(ranked[:k]) if item in expected); idcg=sum(1/np.log2(i+2) for i in range(min(k,len(expected))))
    return dcg/idcg if idcg else 0.0
