import io, re, csv, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import chardet
from rapidfuzz.fuzz import ratio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ACCEPTED={'.csv','.tsv','.txt','.xlsx','.xls'}
ROLE_NAMES=['Automatic','Business Name','Category','City','State','Rating','Review Count','Website','Phone','Email','Social','Previous Contact','Previous Response','Lead Source','Business Age','Text','Review Text','Identifier','Numeric','Categorical','Boolean','Datetime']
ALIASES={'Business Name':['business name','company','company name','name'], 'Category':['category','industry','sector'], 'City':['city','town'], 'State':['state','province'], 'Rating':['rating','stars'], 'Review Count':['review count','reviews count','num reviews'], 'Website':['website','url','web'], 'Phone':['phone','mobile','telephone'], 'Email':['email','email address'], 'Social':['social','social media presence'], 'Previous Contact':['previous contact','contacted'], 'Previous Response':['previous response','responded'], 'Lead Source':['lead source','source'], 'Business Age':['business age','years operating'], 'Review Text':['review text','customer review'], 'Text':['description','text','notes','comment']}
MISSING = {'', ' ', 'n/a', 'na', 'null', 'none', '-', '?', '#n/a', '<na>', 'nan'}

def unique_names(values):
    used=set(); names=[]
    for i,v in enumerate(values):
        s=re.sub(r'\s+',' ',str(v).strip()) if pd.notna(v) else f'Column {i+1}'
        if not s or s.lower().startswith('unnamed'): s=f'Column {i+1}'
        base=s; k=2
        while s in used: s=f'{base} {k}'; k+=1
        used.add(s); names.append(s)
    return names

def parse_file(content, filename, sheet=None):
    ext=Path(filename).suffix.lower()
    if ext not in ACCEPTED: raise ValueError('Unsupported file. Accepted types: .csv, .tsv, .txt, .xlsx, .xls.')
    if len(content)>10*1024*1024: raise ValueError('File exceeds the 10 MB limit.')
    notices=[]; sheets=[]
    try:
        if ext in {'.xlsx','.xls'}:
            excel=pd.ExcelFile(io.BytesIO(content),engine='openpyxl' if ext=='.xlsx' else 'xlrd'); sheets=excel.sheet_names
            chosen=sheet or sheets[0]
            if chosen not in sheets: raise ValueError('Selected sheet does not exist.')
            raw=pd.read_excel(excel,sheet_name=chosen,header=None,nrows=100002)
            # Prefer the earliest row with near-maximal occupancy; merged title rows do not qualify.
            first=raw.head(25); counts=first.notna().sum(axis=1); threshold=max(2, counts.max()*.7)
            candidates=counts[counts>=threshold]
            header=int(candidates.index[0]) if len(candidates) else 0
            labels=raw.iloc[header].tolist()
            if header>0:
                parent=raw.iloc[header-1].ffill()
                labels=[v if pd.notna(v) else parent.iloc[i] for i,v in enumerate(labels)]
            df=raw.iloc[header+1:].copy(); df.columns=unique_names(labels)
            if header: notices.append(f'Header detected on row {header+1}; preceding title rows were excluded.')
        else:
            encoding=chardet.detect(content[:100000]).get('encoding') or 'utf-8'
            text=None
            for enc in list(dict.fromkeys(['utf-8-sig',encoding,'utf-8','latin-1'])):
                try: text=content.decode(enc); break
                except (UnicodeDecodeError,LookupError): pass
            if text is None or '\x00' in text: raise ValueError('This does not appear to be a delimited text file.')
            try: delimiter=csv.Sniffer().sniff(text[:20000],delimiters=',\t;|').delimiter
            except csv.Error:
                first=text.splitlines()[0] if text.splitlines() else ''
                delimiter=max([',','\t',';','|'],key=first.count)
            rows=list(csv.reader(io.StringIO(text),delimiter=delimiter))
            if not rows: raise ValueError('File is empty.')
            width=len(rows[0]); malformed=[i+1 for i,r in enumerate(rows[1:],1) if len(r)!=width and any(r)]
            if any(len(r)>width for r in rows[1:]):
                raise ValueError(f'Ragged rows have more fields than the header (rows {malformed[:8]}). Fix these rows and upload again; no records were silently discarded.')
            if malformed: notices.append(f'{len(malformed)} short rows were padded with missing values (first rows: {malformed[:8]}).')
            df=pd.DataFrame([r+[None]*(width-len(r)) for r in rows[1:] if any(r)],columns=unique_names(rows[0]))
            notices.append(f'Detected {enc} encoding and {repr(delimiter)} delimiter.')
        df=df.replace(r'^\s*$',np.nan,regex=True).dropna(how='all').dropna(axis=1,how='all').reset_index(drop=True)
        if len(df)>100000 or len(df.columns)>500: raise ValueError('Maximum size is 100,000 rows and 500 columns.')
        if len(df.columns)<2 or len(df)==0: raise ValueError('Provide a tabular file with at least two columns and one data row.')
        return df,dict(filename=re.sub(r'[^\w. -]','_',Path(filename).name)[:120],sheets=sheets,sheet=sheet or (sheets[0] if sheets else None),notices=notices,hash=hashlib.sha256(content).hexdigest())
    except ValueError: raise
    except Exception as e: raise ValueError(f'Could not parse this {ext} file. Check its contents and file format. {str(e)[:180]}')

def number(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().lower().replace(',','')
    if s in MISSING:return np.nan
    m=re.fullmatch(r'[^\d+\-.]*([+-]?\d*\.?\d+)\s*([kmb%]?)(?:\s*(?:stars?|years?|reviews?))?',s)
    if not m:return np.nan
    return float(m[1])*{'k':1000,'m':1000000,'b':1000000000,'%':.01,'':1}[m[2]]

def profile(df):
    out=[]
    for col in df:
        s=df[col].dropna().astype(str).str.strip(); sample=s.head(500); h=re.sub(r'[_-]',' ',col.lower())
        role='Automatic'; score=0
        for r,aliases in ALIASES.items():
            value=max(ratio(h,a) for a in aliases)
            if value>score and value>=84:role=r;score=value
        if len(sample):
            if sample.str.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$').mean()>.7:role='Email'
            elif sample.str.match(r'^(?:https?://|www\.)').mean()>.7:role='Website'
            elif sample.str.match(r'^\+\d[\d\s()-]{8,}$').mean()>.7:role='Phone'
        n=s.nunique(); vals=set(sample.str.lower()); numeric=sample.map(number).notna().mean() if len(sample) else 0
        if vals and vals.issubset({'yes','no','y','n','true','false','1','0'}):typ='boolean'
        elif role in ['Email','Phone','Website','Business Name'] or re.search(r'(^id$|[_ ]id$|uuid|identifier)',h):typ='identifier'
        elif role in ['Rating','Review Count','Business Age'] or numeric>.85:typ='numeric'
        elif 'date' in h or (len(sample)>0 and sample.str.match(r'^\d{4}[-/]\d{1,2}[-/]').mean()>.7):typ='datetime'
        elif role in ['Text','Review Text'] or (len(sample)>0 and sample.str.len().mean()>65 and n>20):typ='free text'
        elif len(s)>0 and n/len(s)>.98 and n>30:typ='identifier'
        else:typ='categorical'
        out.append(dict(name=col,type=typ,role=role,unique=n,missing=int(df[col].isna().sum()),excluded=typ=='identifier' and role not in ['Email','Phone','Website']))
    return out

def records(df, n=10):
    return df.head(n).astype(object).where(pd.notna(df.head(n)),None).to_dict('records')

def target_info(df, target, positive=None):
    if target not in df: raise ValueError('Choose an existing target column.')
    s=df[target].astype('string').str.strip().str.lower(); mask=s.notna() & ~s.isin(MISSING)
    vals=s[mask].value_counts(); labels=list(vals.index)
    if len(labels)<2:raise ValueError('The target has only one class. Both positive and negative outcomes are required.')
    numeric=pd.to_numeric(s[mask],errors='coerce')
    if numeric.notna().mean()>.95 and (len(labels)>20 or (len(labels)>2 and (numeric.mod(1)!=0).any())):raise ValueError('This target appears continuous. LeadIQ supports binary classification; choose a categorical outcome.')
    if positive is None:
        positive=next((x for x in ['1','yes','true','converted','won','y'] if x in labels),labels[-1])
    positive=str(positive).strip().lower()
    if positive not in labels:raise ValueError('Positive class is not present in the target.')
    y=(s[mask]==positive).astype(int)
    if len(y)<200:raise ValueError(f'Only {len(y)} usable labeled rows. At least 200 are required; 1,000+ is recommended.')
    if y.value_counts().min()<8:raise ValueError('At least 8 examples in each class are required for stratified evaluation and calibration.')
    return dict(target=target,positive=positive,values=labels,counts={str(k):int(v) for k,v in vals.items()},labeled=len(y),missing_labels=int((~mask).sum()),positives=int(y.sum()),imbalance_ratio=float((1-y).sum()/y.sum())),mask,y

def leakage(df, target, mask, y):
    train,_=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    source=df.loc[mask].iloc[train]; yt=y.iloc[train]; flags=[]
    for c in df:
        if c==target:continue
        reasons=[]
        if re.search(r'converted|conversion|outcome|status|closed|won|deal|result',c,re.I):reasons.append('Outcome-like column name')
        num=source[c].map(number)
        valid=num.notna()
        if valid.sum()>20 and yt[valid].nunique()==2 and num[valid].nunique()>1:
            auc=float(roc_auc_score(yt[valid],num[valid])); corr=float(np.corrcoef(yt[valid],num[valid])[0,1])
            if max(auc,1-auc)>.95:reasons.append(f'Single-feature training ROC-AUC {max(auc,1-auc):.3f} > 0.95')
            if abs(corr)>.9:reasons.append(f'Training correlation {corr:.3f} exceeds ±0.9')
        else:
            cat=source[c].astype('string'); values=cat.dropna().unique()
            if len(values)==2:
                encoded=(cat==values[0]).fillna(False).astype(float); auc=roc_auc_score(yt,encoded)
                if max(auc,1-auc)>.95:reasons.append(f'Binary single-feature training ROC-AUC {max(auc,1-auc):.3f} > 0.95')
        if reasons:flags.append({'column':c,'reason':'; '.join(reasons),'excluded':True})
    return flags