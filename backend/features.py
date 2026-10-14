import time, re, functools, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,TargetEncoder
from sklearn.decomposition import PCA
from storage import folder,read,save,complete,invalidate
from dataio import number,MISSING
from cleaning import structural

class SkewLog(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        arr=np.asarray(X,dtype=float); self.skew_=np.abs(pd.DataFrame(arr).skew().fillna(0).values)>2;return self
    def transform(self,X):
        arr=np.asarray(X,dtype=float).copy();arr[:,self.skew_]=np.sign(arr[:,self.skew_])*np.log1p(np.abs(arr[:,self.skew_]));return arr
    def get_feature_names_out(self,input_features=None):return np.asarray(input_features,dtype=object)

@functools.lru_cache(maxsize=2)
def load_text_model(kind):
    import torch
    torch.set_num_threads(2)
    if kind=='embedding':
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',trust_remote_code=False)
    from transformers import pipeline
    return pipeline('sentiment-analysis',model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',device=-1)

def engineer(df, spec, config, log=None, initial=False):
    out=pd.DataFrame(index=df.index); statuses=[]
    for c in spec['required']:
        role=config['roles'].get(c,'Automatic');typ=config['types'].get(c,'categorical');s=df[c]
        if role in ['Website','Phone','Email','Social']:
            out[c+' presence']=(s.notna()&~s.astype(str).str.lower().isin(['0','false','no','nan','none',''])).astype(float)
            flag=c+' is_valid'
            if flag in df:out[flag]=pd.to_numeric(df[flag],errors='coerce')
        elif typ=='numeric' or role in ['Numeric','Rating','Review Count','Business Age']:out[c]=s.map(number)
        elif typ=='boolean' or role=='Boolean':
            def to_bool_num(v):
                if pd.isna(v): return np.nan
                sv = str(v).strip().lower()
                if sv in ['yes','true','y','1']: return 1.0
                if sv in ['no','false','n','0']: return 0.0
                return np.nan
            out[c]=s.map(to_bool_num)
        elif typ=='datetime' or role=='Datetime':
            dt=pd.to_datetime(s,errors='coerce',utc=True,format='mixed');out[c+' year']=dt.dt.year;out[c+' month']=dt.dt.month;out[c+' weekday']=dt.dt.weekday;out[c+' days since']=(pd.Timestamp(spec['reference_date'])-dt).dt.total_seconds()/86400
        elif typ=='free text' or role in ['Text','Review Text']:
            if not spec['pretrained']:continue
            kind='embedding';started=time.perf_counter();status={'column':c,'model':'sentence-transformers/all-MiniLM-L6-v2'}
            try:
                if not initial and c not in spec.get('text_success',[]):continue
                if log:log(30,f'Loading local CPU embeddings for {c}')
                model=load_text_model(kind);texts=s.fillna('').astype(str).tolist();vectors=model.encode(texts,batch_size=64,show_progress_bar=False,normalize_embeddings=True)
                emb=pd.DataFrame(vectors,index=df.index,columns=[f'{c} embedding {i}' for i in range(vectors.shape[1])]);out=pd.concat([out,emb],axis=1)
                status.update(status='available',truncated_records=sum(len(model.tokenizer.encode(t,add_special_tokens=True))>model.max_seq_length for t in texts),input_token_limit=model.max_seq_length)
                if initial:spec.setdefault('text_success',[]).append(c)
            except Exception as e:
                if not initial:raise ValueError(f'The saved text pipeline cannot load its required embedding model: {str(e)[:200]}')
                status.update(status='unavailable',notice=f'Text features omitted: {str(e)[:240]}')
            status['seconds']=round(time.perf_counter()-started,2);statuses.append(status)
            if role=='Review Text' and (initial or c in spec.get('sentiment_success',[])):
                started=time.perf_counter();ss={'column':c,'model':'distilbert-base-uncased-finetuned-sst-2-english'}
                try:
                    model=load_text_model('sentiment');texts=s.fillna('').astype(str).tolist();pred=model(texts,batch_size=32,truncation=True,max_length=512)
                    out[c+' sentiment']=[v['score'] if v['label']=='POSITIVE' else 1-v['score'] for v in pred];ss.update(status='available',truncated_records=sum(len(model.tokenizer.encode(t))>512 for t in texts),input_token_limit=512)
                    if initial:spec.setdefault('sentiment_success',[]).append(c)
                except Exception as e:
                    if not initial:raise ValueError('Required sentiment model could not be loaded for this saved pipeline.')
                    ss.update(status='unavailable',notice=f'Sentiment omitted: {str(e)[:240]}')
                ss['seconds']=round(time.perf_counter()-started,2);statuses.append(ss)
        else:out[c]=s.map(lambda v:np.nan if pd.isna(v) else str(v))
    return out,statuses

def preprocessor(X):
    embedding=[c for c in X if ' embedding ' in c];num=[c for c in X.select_dtypes(include='number') if c not in embedding];cat=[c for c in X if c not in embedding+num]
    low=[c for c in cat if X[c].nunique()<=20];high=[c for c in cat if c not in low];parts=[];groups={}
    if num:parts.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('log',SkewLog()),('scale',StandardScaler())]),num));groups['numeric']=num
    if low:parts.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=.01,sparse_output=False))]),low));groups['categorical']=low
    if high:parts.append(('target_encoded',TargetEncoder(target_type='binary',smooth='auto',cv=3,shuffle=True,random_state=42),high));groups['cross_fitted_target_encoding']=high
    if embedding:
        for source in sorted(set(c.split(' embedding ')[0] for c in embedding)):
            cols=[c for c in embedding if c.startswith(source+' embedding ')];dims=min(8,len(X),len(cols));parts.append((f'embedding{len(parts)}',PCA(n_components=dims,random_state=42),cols));groups[source+' embedding PCA']=[f'Dimension {i+1}' for i in range(dims)]
    if not parts:raise ValueError('No usable features remain. Include valid numeric or categorical columns, or enable available text features.')
    return ColumnTransformer(parts,remainder='drop',verbose_feature_names_out=True),groups

def feature_job(pid, pretrained,log):
    cfg=read(pid,'config');df=pd.read_pickle(folder(pid)/'clean.pkl');log(5,'Reserving the original stratified holdout; it remains untouched for model selection')
    valid=df[cfg['target']].notna()&~df[cfg['target']].astype(str).str.lower().isin(MISSING);df=df.loc[valid]
    y=(df[cfg['target']].astype(str).str.strip().str.lower()==cfg['positive']).astype(int)
    required=[c for c in df if c!=cfg['target'] and c not in cfg['excluded'] and c in cfg['roles']]
    spec={'required':required,'pretrained':pretrained,'reference_date':cfg['reference_date'],'text_success':[],'sentiment_success':[]}
    X,statuses=engineer(df,spec,cfg,log,True)
    train_ids=[i for i in df.index if i in set(cfg['train_ids'])];test_ids=[i for i in df.index if i in set(cfg['test_ids'])]
    if len(df)<200 or y.loc[train_ids].value_counts().min()<6 or y.loc[test_ids].nunique()<2:raise ValueError('Cleaning left too few labeled examples in one class for a reliable train/test split.')
    # Required raw fields only include inputs that actually survived optional text loading.
    spec['required']=[c for c in required if cfg['types'].get(c)!='free text' and cfg['roles'].get(c) not in ['Text','Review Text'] or c in spec['text_success'] or c in spec['sentiment_success']]
    log(65,'Fitting training-only imputers, rare-category encoding, and feature reduction')
    prep,groups=preprocessor(X.loc[train_ids]);prep.fit_transform(X.loc[train_ids],y.loc[train_ids]);names=prep.get_feature_names_out().tolist()
    split={'train':len(train_ids),'test':len(test_ids),'train_positives':int(y.loc[train_ids].sum()),'test_positives':int(y.loc[test_ids].sum()),'train_negative_positive_ratio':float((1-y.loc[train_ids]).sum()/y.loc[train_ids].sum()),'missing_labels_excluded':int((~valid).sum()),'original_split':'Stratified 80/20 before cleaning; removed duplicates preserve original membership.'}
    warnings_list=[]
    if split['train_positives']<30:warnings_list.append('Fewer than 30 training positives: results are unreliable.')
    if len(df)<1000:warnings_list.append('Fewer than 1,000 rows: more representative historical data is recommended.')
    warnings_list.extend(s['notice'] for s in statuses if s['status']=='unavailable')
    if any(s['status']=='available' for s in statuses):warnings_list.append('Pretrained text models are English-first. Long texts are truncated at the displayed token limits; non-English text may be unreliable.')
    joblib.dump({'X':X,'y':y,'train_ids':train_ids,'test_ids':test_ids,'preprocessor':prep,'spec':spec,'config':cfg},folder(pid)/'bundle.joblib')
    save(pid,'features',{'split':split,'groups':groups,'features':names,'count':len(names),'pretrained_models':statuses,'warnings':warnings_list,'required_columns':spec['required']});complete(pid,4);log(100,f'Created {len(names)} features; saved {len(test_ids)} untouched test records')