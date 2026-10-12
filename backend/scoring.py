import numpy as np
import pandas as pd
import joblib
from storage import folder,read,save,complete
from features import engineer
from cleaning import structural

def score_job(pid,scope,log,new_df=None,new_meta=None):
    trained=joblib.load(folder(pid)/'trained.joblib');b=trained['bundle'];cfg=b['config'];spec=b['spec'];drift=[]
    if new_df is not None:
        missing=[c for c in spec['required'] if c not in new_df]
        if missing:raise ValueError('Missing required columns: '+', '.join(missing))
        raw=new_df.copy();clean,_=structural(new_df,cfg);log(15,'Applying the saved structural transforms and pretrained features');X,_=engineer(clean,spec,cfg,log,False)
        missing_features=[c for c in b['X'] if c not in X]
        if missing_features:raise ValueError('Saved pipeline inputs are missing: '+', '.join(missing_features))
        X=X.reindex(columns=b['X'].columns);train=b['X'].loc[b['train_ids']]
        for c in X:
            if ' embedding ' in c:continue
            if pd.api.types.is_numeric_dtype(X[c]):
                low,high=train[c].min(),train[c].max();outside=int(((X[c]<low)|(X[c]>high)).sum())
                if outside:drift.append({'column':c,'kind':'Outside training range','detail':f'{outside} rows outside [{low:.4g}, {high:.4g}]'})
            else:
                unseen=set(X[c].dropna().astype(str))-set(train[c].dropna().astype(str))
                if unseen:drift.append({'column':c,'kind':'Unseen categories','detail':', '.join(sorted(unseen)[:8])})
            diff=float(X[c].isna().mean()-train[c].isna().mean())
            if abs(diff)>.1:drift.append({'column':c,'kind':'Missingness changed','detail':f'{diff:+.1%} versus training'})
        extras=[c for c in raw if c not in spec['required'] and c!=cfg['target']]
        if extras:drift.append({'column':', '.join(extras),'kind':'Extra columns ignored','detail':'Not required by the saved model.'})
        source='new';in_sample=np.zeros(len(raw),dtype=bool)
    else:
        original=pd.read_pickle(folder(pid)/'raw.pkl')
        if scope=='test':
            raw=original.loc[b['test_ids']].copy();X=b['X'].loc[b['test_ids']].copy();in_sample=np.zeros(len(raw),dtype=bool)
        else:
            raw=original.copy();clean,_=structural(raw,cfg);X,_=engineer(clean,spec,cfg,log,False);X=X.reindex(columns=b['X'].columns);in_sample=raw.index.isin(b['train_ids'])
        source=scope
    log(65,f'Scoring {len(X)} records with the saved calibrated model')
    probability=trained['calibrated'].predict_proba(X)[:,1];reference=trained['train_reference'];scores=np.searchsorted(reference,probability,side='right')/len(reference)*100
    tiers=np.select([scores>=90,scores>=70,scores>=40],['Very High','High','Medium'],default='Low')
    # Preserve source names even when uploaded columns collide with generated output fields.
    reserved={'probability','lead_score','tier','contact_order','lead_id','row_position','in_sample','source_row'}
    output=raw.rename(columns={c:f'Input · {c}' for c in raw if c in reserved}).copy().reset_index(drop=True)
    output['source_row']=[int(i)+2 for i in raw.index];output['lead_id']=[f'{source}-{i}' for i in range(len(raw))];output['row_position']=np.arange(len(raw));output['probability']=probability;output['lead_score']=np.round(scores,2);output['tier']=tiers;output['in_sample']=in_sample
    output=output.sort_values(['lead_score','probability','row_position'],ascending=[False,False,True],kind='stable').reset_index(drop=True);output.insert(0,'contact_order',np.arange(1,len(output)+1))
    output.to_pickle(folder(pid)/'scores.pkl');joblib.dump({'X':X,'raw':raw,'config':cfg},folder(pid)/'scored.joblib')
    histogram=[]
    for low in range(0,100,10):histogram.append({'name':f'{low}–{low+10}','count':int(((scores>=low)&(scores<(low+10) if low<90 else scores<=100)).sum()),'tier':'Very High' if low>=90 else 'High' if low>=70 else 'Medium' if low>=40 else 'Low'})
    summary={'total':len(output),'tiers':{t:int((tiers==t).sum()) for t in ['Very High','High','Medium','Low']},'histogram':histogram,'source':source,'in_sample_count':int(in_sample.sum()),'drift':drift,'columns':list(output.columns),'new_file':new_meta}
    save(pid,'score_summary',summary);complete(pid,7);log(100,'Calibrated probabilities, percentile scores, tiers, and contact order saved')

def leads(pid,search='',tier='',page=1,page_size=20,sort='contact_order',descending=False,filter_column='',filter_value='',top_n=None):
    df=pd.read_pickle(folder(pid)/'scores.pkl')
    if top_n:df=df.nsmallest(top_n,'contact_order')
    if tier:df=df[df['tier']==tier]
    if search:
        mask=pd.Series(False,index=df.index)
        for c in df:mask|=df[c].astype(str).str.contains(search,case=False,regex=False,na=False)
        df=df[mask]
    if filter_column and filter_column in df and filter_value:df=df[df[filter_column].astype(str).str.contains(filter_value,case=False,regex=False,na=False)]
    if sort in df:
        try:df=df.sort_values(sort,ascending=not descending,kind='stable')
        except TypeError:df=df.sort_values(sort,key=lambda s:s.astype(str),ascending=not descending,kind='stable')
    total=len(df);s=df.iloc[(page-1)*page_size:page*page_size]
    return {'rows':s.astype(object).where(pd.notna(s),None).to_dict('records'),'total':total,'page':page,'pages':max(1,int(np.ceil(total/page_size))),'columns':list(df.columns)}