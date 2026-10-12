import os, json, uuid, hashlib, secrets, shutil, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from pymongo import MongoClient
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')
ROOT = Path(os.environ['PROJECT_DATA_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
db = MongoClient(os.environ['MONGO_URL'])[os.environ['DB_NAME']]
pool = ThreadPoolExecutor(max_workers=2)
locks = {}

def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items() if k != '_id'}
    if isinstance(x, (list, tuple, np.ndarray)): return [plain(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (datetime, pd.Timestamp)): return x.isoformat()
    if x is pd.NA or x is pd.NaT: return None
    return x

def folder(pid):
    if not isinstance(pid, str) or len(pid) != 32 or any(c not in '0123456789abcdef' for c in pid):
        raise HTTPException(404, 'Project not found.')
    return ROOT / pid

def save(pid, name, obj):
    p = folder(pid) / (name + '.json')
    if not p.parent.exists(): raise ValueError('Project was deleted or expired.')
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(obj), allow_nan=False, default=str))
    tmp.replace(p)

def read(pid, name):
    p = folder(pid) / (name + '.json')
    if not p.exists(): raise HTTPException(409, f'{name.replace("_", " ").title()} is not available yet. Complete the preceding step.')
    return json.loads(p.read_text())

def create():
    pid, token = uuid.uuid4().hex, secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    doc = dict(id=pid, token_hash=hashlib.sha256(token.encode()).hexdigest(), created_at=now.isoformat(), expires_at=(now+timedelta(hours=24)).isoformat(), step=1, completed=[], name='Untitled project', synthetic=False)
    folder(pid).mkdir()
    db.projects.insert_one(doc.copy())
    return {**{k:v for k,v in doc.items() if k!='token_hash'}, 'token':token}

def project(pid, token):
    folder(pid)
    doc = db.projects.find_one({'id':pid}, {'_id':0})
    if not doc or not secrets.compare_digest(doc['token_hash'], hashlib.sha256((token or '').encode()).hexdigest()):
        raise HTTPException(404, 'Project unavailable. It may have expired, or this browser does not have its access key.')
    if datetime.fromisoformat(doc['expires_at']) <= datetime.now(timezone.utc):
        remove(pid)
        raise HTTPException(410, 'This project expired after 24 hours. Please start a new project.')
    return {k:v for k,v in doc.items() if k!='token_hash'}

def update(pid, **values):
    db.projects.update_one({'id':pid}, {'$set':plain(values)})

def complete(pid, step):
    db.projects.update_one({'id':pid}, {'$addToSet':{'completed':step}})

def invalidate(pid, step):
    # Revoke completion before deleting artifacts so concurrent readers cannot use old results.
    revision=uuid.uuid4().hex
    state={'results_revision':revision,'invalidated_from':step,'invalidated_at':datetime.now(timezone.utc).isoformat()}
    if step<=5:state['active_run_id']=None
    db.projects.update_one({'id':pid},{'$pull':{'completed':{'$gte':step}},'$set':state})
    names = {1:['config'],2:['clean','quality','eda'],4:['features','bundle'],5:['metrics','trained','importance','shap_background','verification'],7:['scores','scored','score_summary']}
    for s, files in names.items():
        if s>=step:
            for name in files:
                for ext in ['json','joblib','pkl','npz']:
                    (folder(pid)/f'{name}.{ext}').unlink(missing_ok=True)
    if step<=5:
        for pattern in ['predictions_*.npz','inspection_*.json']:
            for artifact in folder(pid).glob(pattern):artifact.unlink(missing_ok=True)
    # Issued downloads are snapshots of the old result; do not let them outlive invalidation.
    db.downloads.delete_many({'project_id':pid})
    shutil.rmtree(folder(pid)/'downloads',ignore_errors=True)

def remove(pid):
    db.projects.delete_one({'id':pid})
    db.jobs.delete_many({'project_id':pid})
    db.downloads.delete_many({'project_id':pid})
    shutil.rmtree(folder(pid), ignore_errors=True)

def cleanup():
    for p in db.projects.find({'expires_at':{'$lte':datetime.now(timezone.utc).isoformat()}}, {'_id':0,'id':1}): remove(p['id'])
    for record in db.downloads.find({'expires_at':{'$lte':datetime.now(timezone.utc).isoformat()}},{'_id':0}):
        (folder(record['project_id'])/'downloads'/record['stored']).unlink(missing_ok=True)
        db.downloads.delete_one({'token_hash':record['token_hash']})

def start_job(pid, kind, fn):
    lock=locks.setdefault(pid, threading.Lock())
    if not lock.acquire(blocking=False): raise HTTPException(409,'A job is already running for this project.')
    jid=uuid.uuid4().hex
    try:
        step={'clean':2,'features':4,'train':5,'score':7}.get(kind)
        if step:invalidate(pid,step)
    except Exception:
        lock.release();raise
    db.jobs.insert_one(dict(id=jid,project_id=pid,kind=kind,status='queued',progress=0,logs=[],created_at=datetime.now(timezone.utc).isoformat()))
    if kind!='inspection':update(pid, active_job=jid)
    def log(progress, message):
        if not db.projects.find_one({'id':pid}): raise ValueError('Project deleted during processing.')
        db.jobs.update_one({'id':jid},{'$set':{'progress':progress,'stage':message,'status':'running'},'$push':{'logs':{'time':datetime.now(timezone.utc).strftime('%H:%M:%S'),'message':message}}})
    def run():
        try:
            fn(log)
            db.jobs.update_one({'id':jid},{'$set':{'status':'completed','progress':100,'stage':'Complete'}})
        except Exception as e:
            import logging
            logging.exception('Job failed: %s',kind)
            db.jobs.update_one({'id':jid},{'$set':{'status':'failed','error':str(e),'stage':'Could not complete this step'}})
        finally: lock.release()
    pool.submit(run)
    return {'job_id':jid}