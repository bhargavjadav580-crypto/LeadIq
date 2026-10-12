import re
import numpy as np
import pandas as pd
import phonenumbers
from rapidfuzz import fuzz, process
from storage import folder,read,save,complete,invalidate
from dataio import MISSING,number,records,profile

def structural(df, config):
    df=df.copy(); invalid={'email':0,'phone':0}
    for col in df:
        role=config.get('roles',{}).get(col,'Automatic'); typ=config.get('types',{}).get(col)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col]=df[col].map(lambda v: np.nan if pd.isna(v) or str(v).strip().lower() in MISSING else str(v).strip().lower())
        if col==config.get('target'):continue
        if typ=='numeric' or role in ['Numeric','Rating','Review Count','Business Age']:
            df[col]=df[col].map(number)
            if role=='Rating':df.loc[(df[col]<0)|(df[col]>5),col]=np.nan
            if role in ['Review Count','Business Age']:df.loc[df[col]<0,col]=np.nan
        if typ=='boolean' or role=='Boolean':df[col]=df[col].replace({'yes':1,'true':1,'y':1,'no':0,'false':0,'n':0,'1':1,'0':0})
        if typ=='datetime' or role=='Datetime':df[col]=pd.to_datetime(df[col],errors='coerce',format='mixed',utc=True).map(lambda v:v.isoformat() if pd.notna(v) else np.nan)
        if role=='Email':
            valid=df[col].fillna('').astype(str).str.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$'); invalid['email']+=int((df[col].notna()&~valid).sum());df[f'{col} is_valid']=valid.astype(int)
        if role=='Phone':
            def phone(v):
                if pd.isna(v):return None
                try:
                    p=phonenumbers.parse(str(v),'IN')
                    return phonenumbers.format_number(p,phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(p) else None
                except Exception:return None
            normalized=df[col].map(phone);valid=normalized.notna();invalid['phone']+=int((df[col].notna()&~valid).sum());df[f'{col} is_valid']=valid.astype(int);df.loc[valid,col]=normalized[valid]
        if role=='Website':df[col]=df[col].map(lambda v: np.nan if pd.isna(v) else (str(v) if re.match(r'^https?://',str(v)) else 'https://'+str(v)))
    return df,invalid

def clean_job(pid, overrides, log):
    config=read(pid,'config'); raw=pd.read_pickle(folder(pid)/'raw.pkl'); log(8,'Standardizing missing values, text, dates, and numbers')
    df,invalid=structural(raw,config); before=len(df); exact=int(df.duplicated().sum());df=df.drop_duplicates()
    log(32,f'Removed {exact} exact duplicates; validating contact details')
    names=[c for c,r in config['roles'].items() if r=='Business Name' and c in df]
    cities=[c for c,r in config['roles'].items() if r=='City' and c in df]
    phones=[c for c,r in config['roles'].items() if r=='Phone' and c in df]
    drop=set(); conflicts=0
    if names and (cities or phones):
        name=names[0]; ordered=df.assign(_complete=df.notna().sum(axis=1)).sort_values('_complete',ascending=False,kind='stable')
        # Blocking avoids all-pairs comparisons. Numeric suffixes retain distinct business records.
        blocks={}
        for idx,row in ordered.iterrows():
            if pd.isna(row[name]):continue
            keys=[]
            for c in phones+cities:
                if pd.notna(row[c]):keys.append((c,str(row[c])))
            matched=None
            for key in keys:
                candidates=blocks.get(key,{})
                if candidates:
                    match=process.extractOne(str(row[name]),candidates,scorer=fuzz.ratio,score_cutoff=92)
                    if match:
                        other=match[2]
                        nums=re.findall(r'\d+',str(row[name])); nums2=re.findall(r'\d+',str(df.loc[other,name]))
                        if nums==nums2:matched=other;break
            if matched is not None:
                if str(row[config['target']])!=str(df.loc[matched,config['target']]):conflicts+=1
                drop.add(idx)
            else:
                for key in keys:blocks.setdefault(key,{})[idx]=str(row[name])
    df=df.drop(index=list(drop)); log(65,f'Removed {len(drop)} fuzzy duplicates; auditing columns')
    candidates=[]
    for c in df:
        if c==config['target']:continue
        missing=float(df[c].isna().mean());constant=df[c].nunique(dropna=True)<=1
        if missing>.6 or constant:candidates.append({'column':c,'reason':f'{missing:.1%} missing' if missing>.6 else 'Constant column','dropped':c not in overrides})
    drops=[x['column'] for x in candidates if x['dropped']]
    report=dict(rows_before=before,rows_after=len(df),exact_duplicates=exact,fuzzy_duplicates=len(drop),conflicting_duplicates=conflicts,invalid_emails=invalid['email'],invalid_phones=invalid['phone'],columns_dropped=candidates,missingness=[{'column':c,'before':int(raw[c].isna().sum()),'after':int(df[c].isna().sum()) if c in df else 0} for c in raw],before_sample=records(raw),after_sample=records(df),notices=['Conflicting duplicate outcomes were resolved by retaining the most complete record.'] if conflicts else [])
    if len(df)<200:raise ValueError(f'Only {len(df)} unique rows remain after cleaning; at least 200 are needed.')
    df=df.drop(columns=drops);df.to_pickle(folder(pid)/'clean.pkl');save(pid,'quality',report);complete(pid,2);log(100,'Cleaning complete; quality report saved')

def eda(pid):
    df=pd.read_pickle(folder(pid)/'clean.pkl');cfg=read(pid,'config');target=cfg['target'];valid=df[target].notna();y=(df.loc[valid,target].astype(str).str.lower()==cfg['positive']).astype(int);data=df.loc[valid].copy();cats=[];nums=[]
    for p in profile(data.drop(columns=[target])):
        c=p['name']
        if p['type'] in ['categorical','boolean'] and len(cats)<3:
            temp=pd.DataFrame({'category':data[c].fillna('Missing').astype(str),'y':y});g=temp.groupby('category')['y'].agg(['mean','count']).sort_values('count',ascending=False).head(8)
            vals=[{'name':str(k),'rate':float(v['mean']),'count':int(v['count'])} for k,v in g.iterrows()]
            cats.append({'column':c,'data':vals,'insight':f'{vals[0]["name"]} is the largest group ({vals[0]["count"]} rows), with a {vals[0]["rate"]:.1%} positive outcome rate.' if vals else ''})
        if p['type']=='numeric' and len(nums)<4:
            s=data[c].map(number).dropna()
            if len(s) and s.nunique()>1:
                logscale=bool(s.skew()>2 and s.min()>=0);v=np.log1p(s) if logscale else s;counts,edges=np.histogram(v,bins=12)
                nums.append({'column':c,'log_scale':logscale,'data':[{'name':f'{edges[i]:.1f}–{edges[i+1]:.1f}','count':int(n)} for i,n in enumerate(counts)],'insight':f'Median {s.median():.2f}; {len(s)} nonmissing observations.'+(' Axis shows log(1 + value) because the distribution is skewed.' if logscale else '')})
    numeric=data.select_dtypes(include='number').iloc[:,:12].copy();numeric['Target outcome']=y;corr=numeric.corr().fillna(0)
    result={'balance':[{'name':'Positive','count':int(y.sum())},{'name':'Negative','count':int((1-y).sum())}], 'prevalence':float(y.mean()),'categorical':cats,'numeric':nums,'missingness':[{'name':c,'count':int(data[c].isna().sum())} for c in data], 'correlation':{'columns':list(corr.columns),'values':corr.values.tolist()}}
    save(pid,'eda',result);complete(pid,3);return result