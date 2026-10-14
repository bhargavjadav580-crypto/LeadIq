import importlib.metadata, time
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve,precision_recall_curve,brier_score_loss
from sklearn.calibration import calibration_curve

def evaluate(y,p,threshold=.5):
    p=np.clip(np.asarray(p,dtype=float),0.0,1.0)
    pred=p>=threshold;y=np.asarray(y,dtype=int);n=max(1,int(np.ceil(len(y)*.1)));top=np.argsort(-p,kind='stable')[:n];precision_top=float(y[top].mean());prevalence=float(y.mean())
    undefined={}
    if not pred.any():undefined['precision']='No records are predicted positive at this threshold; sklearn reports 0 by convention.'
    if not y.sum():undefined['recall']='There are no actual positive records; sklearn reports 0 by convention.'
    if not y.sum() and not pred.any():undefined['f1']='There are no actual or predicted positives; sklearn reports 0 by convention.'
    
    brier=float(brier_score_loss(y,p)) if len(y) else None
    majority_acc=float(max(prevalence,1.0-prevalence)) if len(y) else None
    baseline_brier=float(brier_score_loss(y,np.full_like(y,prevalence,dtype=float))) if len(y) else None
    
    return dict(
        accuracy=accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0),
        roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else None,
        pr_auc=average_precision_score(y,p) if y.sum() else None,
        brier_score=brier,
        majority_baseline_accuracy=majority_acc,
        constant_baseline_brier=baseline_brier,
        predicted_conversion_rate=float(pred.mean()),
        actual_conversion_rate=prevalence,
        precision_top10=precision_top,
        lift_top10=precision_top/prevalence if prevalence else None,
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
        threshold=float(threshold),
        undefined=undefined,
        predicted_positives=int(pred.sum()),
        actual_positives=int(y.sum()),
        evaluation_count=len(y)
    )

def thin(x,y,limit=160):
    idx=np.unique(np.linspace(0,len(x)-1,min(limit,len(x))).astype(int));return [{'x':float(x[i]),'y':float(y[i])} for i in idx]

def curves(y,p):
    fpr,tpr,_=roc_curve(y,p);pr,rec,_=precision_recall_curve(y,p)
    return {'roc':thin(fpr,tpr),'pr':thin(rec[::-1],pr[::-1])}

def reliability(y,p):
    observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile');return [{'x':float(a),'y':float(b)} for a,b in zip(predicted,observed)]

def bootstrap(y,p,threshold,log):
    rng=np.random.default_rng(42);y=np.asarray(y);p=np.asarray(p);values={k:[] for k in ['roc_auc','pr_auc','f1']};skipped=0
    for i in range(1000):
        ix=rng.integers(0,len(y),len(y));a,b=y[ix],p[ix]
        if len(np.unique(a))<2:skipped+=1;continue
        values['roc_auc'].append(roc_auc_score(a,b));values['pr_auc'].append(average_precision_score(a,b));values['f1'].append(f1_score(a,b>=threshold,zero_division=0))
        if i in [249,499,749]:log(78+i//100,f'Bootstrap confidence intervals: {i+1}/1,000 resamples computed')
    return {k:{'lower':float(np.percentile(v,2.5)),'upper':float(np.percentile(v,97.5)),'valid_resamples':len(v)} if v else {'lower':None,'upper':None,'valid_resamples':0} for k,v in values.items()},skipped

def gain(y,p):
    order=np.argsort(-np.asarray(p),kind='stable');ys=np.asarray(y)[order];cum=np.cumsum(ys);total=max(1,int(ys.sum()));n=len(ys);points=[{'x':0.,'y':0.}];deciles=[]
    for d in range(1,11):
        lo=int(np.ceil(n*(d-1)/10));hi=int(np.ceil(n*d/10));group=ys[lo:hi];precision=float(group.mean()) if len(group) else 0
        points.append({'x':hi/n,'y':float(cum[hi-1]/total)});deciles.append({'decile':d,'leads':len(group),'positives':int(group.sum()),'precision':precision,'lift':precision/(total/n),'cumulative_gain':float(cum[hi-1]/total)})
    return {'curve':points,'deciles':deciles}

def versions():
    result={}
    for p in ['pandas','numpy','scikit-learn','xgboost-cpu','shap','joblib','sentence-transformers','transformers','openpyxl','xlrd','rapidfuzz','phonenumbers']:
        try:result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:result[p]='unavailable'
    return result