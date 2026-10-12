import numpy as np
import pandas as pd

def sample_data():
    r=np.random.default_rng(42); n=5825
    category=r.choice(['Restaurant','Healthcare','Retail','Education','Home services','Fitness','Professional services'],n)
    city=r.choice(['Mumbai','Delhi','Bengaluru','Pune','Hyderabad','Chennai','Jaipur','Ahmedabad'],n)
    states={'Mumbai':'Maharashtra','Delhi':'Delhi','Bengaluru':'Karnataka','Pune':'Maharashtra','Hyderabad':'Telangana','Chennai':'Tamil Nadu','Jaipur':'Rajasthan','Ahmedabad':'Gujarat'}
    rating=np.clip(r.normal(3.7,.7,n),1,5); reviews=r.lognormal(3.8,1.2,n).astype(int)
    web=r.random(n)<.6; contact=r.random(n)<.35; response=(r.random(n)<.35)&contact; age=r.integers(1,26,n)
    z=-4.65 + 1.4*(rating-3.5)+.7*np.log1p(reviews/40)+2.2*response+.75*web+.35*(category=='Professional services')+r.normal(0,.7,n)
    y=r.random(n)<1/(1+np.exp(-z))
    # Flip 6% of labels, balanced across original classes so the low prevalence is preserved.
    # This is synthetic data generation only; scores always come from fitted classifiers.
    flip_count=min(int(n*.03),int(y.sum()),int((~y).sum()))
    flip=np.concatenate([r.choice(np.flatnonzero(y),flip_count,replace=False),r.choice(np.flatnonzero(~y),flip_count,replace=False)])
    y[flip]=~y[flip]
    df=pd.DataFrame({'Business Name':[f'{r.choice(["Shree","Urban","Lotus","Apex","Green","New","Sai","Metro"])} {category[i]} {i+1}' for i in range(n)],'Category':category,'City':city,'State':[states[c] for c in city], 'Rating':[f'{v:.1f} stars' if i%7==0 else f'{v:.1f}' for i,v in enumerate(rating)],'Review Count':[f'{v:,}' for v in reviews],'Website':[f'https://business{i}.in' if web[i] else None for i in range(n)],'Phone':[f'+9198{i:08d}' if i%19 else '12345' for i in range(n)],'Email':[f'hello@business{i}.in' if i%17 else 'bad@email' for i in range(n)],'Social Media Presence':r.choice(['Yes','No'],n),'Previous Contact':np.where(contact,'Yes','No'),'Previous Response':np.where(response,'Yes','No'),'Lead Source':r.choice(['Google Maps','Referral','Directory','Website'],n),'Business Age':age,'Description':[f'Local {c.lower()} offering services in {ci}. {r.choice(["Family-owned establishment.","Online appointments available.","Serving the local community."])}' for c,ci in zip(category,city)],'Converted':np.where(y,'Yes','No')})
    for c in df.columns:
        if c!='Converted':df.loc[r.random(n)<.05,c]=None
    duplicates=df.iloc[:175].copy()
    duplicates.loc[duplicates.index[::2],'Business Name']=duplicates.loc[duplicates.index[::2],'Business Name'].astype(str)+'  '
    duplicates.loc[duplicates.index[1::4],'Business Name']=duplicates.loc[duplicates.index[1::4],'Business Name'].astype(str)+'.'
    return pd.concat([df,duplicates],ignore_index=True)