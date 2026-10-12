import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from storage import folder,save,read

def explain_values(pipe,background,X):
    import shap
    model=pipe.named_steps['model'];prep=pipe.named_steps['features'];z=prep.transform(X);bg=prep.transform(background)
    if hasattr(model,'coef_'):
        exp=shap.LinearExplainer(model,bg);values=exp(z);scale='uncalibrated log-odds'
    elif model.__class__.__name__=='DummyClassifier':
        raise ValueError('The class-prior Dummy baseline does not use feature values. SHAP is not supported for this constant estimator; no substitute importance is generated.')
    else:
        exp=shap.TreeExplainer(model);values=exp(z);scale='uncalibrated log-odds' if model.__class__.__name__=='XGBClassifier' else 'uncalibrated positive-class probability'
    vals=values.values;base=values.base_values
    if vals.ndim==3:vals=vals[:,:,1];base=base[:,1]
    return vals,base,z,prep.get_feature_names_out().tolist(),scale

def global_importance(pid,pipe,Xtrain,Xtest,ytest,log,artifact_name='importance',save_background=True):
    bg=Xtrain.sample(min(80,len(Xtrain)),random_state=42);sample=Xtest.sample(min(150,len(Xtest)),random_state=42)
    if save_background:joblib.dump(bg,folder(pid)/'shap_background.joblib')
    result={'sample_size':len(sample),'background_size':len(bg),'permutation_sample_size':min(400,len(Xtest)),'permutation_repeats':3,'seed':42,'shap':[],'permutation':[]}
    try:
        log(90,f'Computing real SHAP values for {len(sample)} held-out rows')
        vals,base,z,names,scale=explain_values(pipe,bg,sample);means=np.abs(vals).mean(axis=0);order=np.argsort(-means)[:25];result.update(shap=[{'feature':names[i],'importance':float(means[i])} for i in order],output=scale,status='available')
    except Exception as e:result.update(status='unavailable',notice=f'SHAP could not explain this estimator: {str(e)[:300]}. Predictions remain available.')
    try:
        log(93,'Cross-checking feature influence using permutation importance')
        p=permutation_importance(pipe,Xtest,ytest,scoring='average_precision',n_repeats=3,random_state=42,max_samples=min(400,len(Xtest)),n_jobs=1)
        ix=np.argsort(-p.importances_mean)[:25];result['permutation']=[{'feature':str(Xtest.columns[i]),'importance':float(p.importances_mean[i]),'std':float(p.importances_std[i])} for i in ix]
        if result['shap']:
            tops=[s['feature'].split('__',1)[-1].lower() for s in result['shap'][:5]];perm=[s['feature'].lower() for s in result['permutation'][:5]];overlap=sum(any(a.startswith(b) for a in tops) for b in perm);result['disagreement']=overlap<2
    except Exception as e:result['permutation_notice']=str(e)[:250]
    if artifact_name:save(pid,artifact_name,result)
    return result

def explain_lead(pid,lead_id):
    scores=pd.read_pickle(folder(pid)/'scores.pkl');rows=scores[scores['lead_id']==lead_id]
    if rows.empty:raise ValueError('Lead not found in the current ranked dataset.')
    row=rows.iloc[0];trained=joblib.load(folder(pid)/'trained.joblib');scored=joblib.load(folder(pid)/'scored.joblib');X=scored['X'].iloc[[int(row['row_position'])]];bg=joblib.load(folder(pid)/'shap_background.joblib')
    vals,base,z,names,scale=explain_values(trained['best_pipeline'],bg,X);v=vals[0];pos=sorted([i for i in range(len(v)) if v[i]>0],key=lambda i:-v[i])[:5];neg=sorted([i for i in range(len(v)) if v[i]<0],key=lambda i:v[i])[:5]
    original=scored['raw'].iloc[int(row['row_position'])].to_dict()
    contributions=[]
    for i in pos+neg:
        feature=names[i];short=feature.split('__',1)[-1];source=next((c for c in sorted(original,key=len,reverse=True) if short.startswith(c)),None)
        contributions.append({'feature':feature,'value':float(v[i]),'transformed_value':float(z[0,i]),'source_column':source,'actual_value':str(original.get(source,'Derived feature'))})
    reasons=[f'{c["feature"].split("__",1)[-1]} ({c["actual_value"]}) contributes {c["value"]:+.4f}' for c in contributions[:3]]
    name=next((str(original[c]) for c,r in scored['config']['roles'].items() if r=='Business Name' and c in original),'there')
    outreach=f'Hello {name}, I would love to learn more about your current priorities and whether our services could be useful. Would you be open to a brief conversation?'
    if row['tier'] in ['Very High','High']:outreach=f'Hello {name}, I’m reaching out to explore whether we could help with your current goals. Would a short conversation this week be convenient?'
    return {'lead_id':lead_id,'score':float(row['lead_score']),'probability':float(row['probability']),'tier':row['tier'],'contributions':contributions,'base_value':float(np.asarray(base).reshape(-1)[0]),'output':scale,'all_contributions_sum':float(v.sum()),'text':'Text generated from SHAP values: '+('; '.join(reasons) if reasons else 'No nonzero contributions were found.')+f' to the {scale}. These are model-output contributions, not score points.','outreach':outreach,'outreach_rationale':f'{row["tier"]} priority. Model reasons for internal preparation only: '+('; '.join(reasons[:2]) or 'No nonzero reasons.'),'record':original,'notice':'SHAP explains the underlying uncalibrated estimator, not the calibrated probability, percentile score, or causation.'}