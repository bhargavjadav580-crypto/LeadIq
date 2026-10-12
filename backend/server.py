import io,os,re,json,secrets,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent/'.env')
from fastapi import FastAPI,APIRouter,HTTPException,Header,Depends,UploadFile,File,Form,BackgroundTasks,Request,Query
from fastapi.responses import Response,JSONResponse,FileResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from storage import db,folder,plain,create,project,update,remove,read,save,start_job,complete,invalidate,cleanup,locks
from dataio import parse_file,profile,records,target_info,leakage,ROLE_NAMES
from sample import sample_data
from cleaning import clean_job,eda
from features import feature_job
from training import train_job,verify
from scoring import score_job,leads
from explanations import explain_lead
from exports import tabular,pdf_export,methodology
from evaluation import evaluate
from inspections import current_run,prediction_set,inspect_model,ensure_analysis

@asynccontextmanager
async def lifespan(app):
    db.projects.create_index('id',unique=True);db.jobs.create_index('id',unique=True);db.cron_runs.create_index('run_id',unique=True);db.cron_runs.create_index('created_at',expireAfterSeconds=172800)
    db.jobs.update_many({'status':{'$in':['running','queued']}},{'$set':{'status':'failed','error':'Processing was interrupted by a service restart. Please rerun this step.'}})
    cleanup()
    yield

app=FastAPI(title='LeadIQ',lifespan=lifespan);router=APIRouter(prefix='/api')
origins=os.environ['CORS_ORIGINS'].split(',')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=['GET','POST','PATCH','DELETE'],allow_headers=['Content-Type','X-Project-Key','Authorization','X-Webhook-Id','X-Webhook-Timestamp'])

@app.exception_handler(ValueError)
async def value_error(request,exc):return JSONResponse(status_code=400,content={'detail':str(exc)})
@app.exception_handler(FileNotFoundError)
async def missing_file(request,exc):return JSONResponse(status_code=409,content={'detail':'This result is not available yet. Complete the preceding step.'})

def auth(pid:str,x_project_key:str=Header(default='')):return project(pid,x_project_key)
def idle(pid):
    if locks.get(pid) and locks[pid].locked():raise HTTPException(409,'Wait for the running job before changing this project.')
def send(data,mime,name):return Response(data,media_type=mime,headers={'Content-Disposition':f'attachment; filename="{name}"','Cache-Control':'no-store'})
def require(pid,step):
    p=db.projects.find_one({'id':pid},{'_id':0,'completed':1})
    if not p or step not in p['completed']:raise HTTPException(409,f'Complete step {step} first.')

class Configure(BaseModel):
    target:str
    positive:str|None=None
    roles:dict[str,str]=Field(default_factory=dict)
    excluded:list[str]=Field(default_factory=list)
    leakage_overrides:list[str]=Field(default_factory=list)
class Clean(BaseModel):keep_columns:list[str]=Field(default_factory=list)
class FeatureOptions(BaseModel):pretrained:bool=True
class TrainOptions(BaseModel):mode:str='quick'
class ScoreOptions(BaseModel):scope:str='test'
class Sheet(BaseModel):sheet:str
class Step(BaseModel):step:int=Field(ge=1,le=9)
class Threshold(BaseModel):
    model:str
    threshold:float=Field(ge=0,le=1)
    run_id:str|None=None
class InspectRequest(BaseModel):
    model:str
    run_id:str
class DownloadRequest(BaseModel):
    kind:str='export'
    format:str='pdf'
    top_n:int=Field(default=25,ge=1,le=1000)
    inline:bool=False

@router.get('/health')
def health():db.command('ping');return {'status':'ok','service':'LeadIQ'}
@router.post('/projects',response_model=dict)
def create_project():return create()
@router.get('/projects/{pid}',response_model=dict)
def get_project(pid:str,p=Depends(auth)):return p
@router.delete('/projects/{pid}')
def delete_project(pid:str,p=Depends(auth)):remove(pid);return {'deleted':True}
@router.patch('/projects/{pid}')
def change_step(pid:str,body:Step,p=Depends(auth)):
    prereq={2:1,3:2,4:2,5:4,6:5,7:5,8:7,9:7}
    if body.step in prereq:require(pid,prereq[body.step])
    update(pid,step=body.step);return {'step':body.step}

def store_upload(pid,content,filename,sheet=None,synthetic=False):
    df,meta=parse_file(content,filename,sheet);invalidate(pid,1);df.to_pickle(folder(pid)/'raw.pkl');(folder(pid)/'source.bin').write_bytes(content);meta['synthetic']=synthetic;save(pid,'upload',meta);update(pid,name=meta['filename'],synthetic=synthetic,step=1,active_job=None);return {'rows':len(df),'columns':len(df.columns),**meta}

@router.post('/projects/{pid}/upload')
async def upload(pid:str,file:UploadFile=File(...),p=Depends(auth)):
    idle(pid);content=await file.read(10*1024*1024+1);return store_upload(pid,content,file.filename or '')
@router.post('/projects/{pid}/sample')
def use_sample(pid:str,p=Depends(auth)):
    idle(pid);return store_upload(pid,sample_data().to_csv(index=False).encode(),'synthetic-business-leads.csv',synthetic=True)
@router.post('/projects/{pid}/sheet')
def select_sheet(pid:str,body:Sheet,p=Depends(auth)):
    idle(pid);meta=read(pid,'upload');return store_upload(pid,(folder(pid)/'source.bin').read_bytes(),meta['filename'],body.sheet,meta['synthetic'])
@router.get('/projects/{pid}/preview')
def preview(pid:str,p=Depends(auth)):
    meta=read(pid,'upload');df=pd.read_pickle(folder(pid)/'raw.pkl');columns=profile(df);values={c:df[c].dropna().astype(str).str.strip().unique()[:500].tolist() for c in df};suggested=next((c for c in df if re.search(r'converted|outcome|target|label|subscribed|^y$',c,re.I)),None)
    cfg=read(pid,'config') if (folder(pid)/'config.json').exists() else None
    return plain({**meta,'rows':len(df),'columns':columns,'preview':records(df),'values':values,'suggested_target':suggested,'config':cfg})
@router.post('/projects/{pid}/configure')
def configure(pid:str,body:Configure,p=Depends(auth)):
    idle(pid);df=pd.read_pickle(folder(pid)/'raw.pkl');info,mask,y=target_info(df,body.target,body.positive);flags=leakage(df,body.target,mask,y);profiles=profile(df);roles={r['name']:body.roles.get(r['name'],r['role']) for r in profiles};types={r['name']:r['type'] for r in profiles}
    for c,r in roles.items():
        if r not in ROLE_NAMES:raise ValueError(f'Invalid column role: {r}')
        if r in ['Numeric','Categorical','Boolean','Datetime','Identifier']:types[c]=r.lower()
        if r in ['Text','Review Text']:types[c]='free text'
    default_excluded={r['name'] for r in profiles if r['excluded']} if 'excluded' not in body.model_fields_set else set()
    excluded=set(body.excluded)|default_excluded|{f['column'] for f in flags if f['column'] not in body.leakage_overrides}|{c for c,r in roles.items() if r=='Identifier'}
    train,test=train_test_split(df.index[mask].tolist(),test_size=.2,stratify=y,random_state=42)
    for f in flags:f['excluded']=f['column'] in excluded
    config={**info,'roles':roles,'types':types,'excluded':sorted(excluded),'leakage':flags,'train_ids':train,'test_ids':test,'reference_date':p['created_at']};invalidate(pid,1);save(pid,'config',config);complete(pid,1);return plain(config)

@router.post('/projects/{pid}/clean')
def cleaning(pid:str,body:Clean,p=Depends(auth)):idle(pid);require(pid,1);return start_job(pid,'clean',lambda log:clean_job(pid,body.keep_columns,log))
@router.get('/projects/{pid}/quality')
def quality(pid:str,p=Depends(auth)):return read(pid,'quality')
@router.get('/projects/{pid}/cleaned')
def cleaned_download(pid:str,format:str='csv',p=Depends(auth)):
    if format not in ['csv','xlsx']:raise ValueError('Choose CSV or Excel.')
    data,mime=tabular(pd.read_pickle(folder(pid)/'clean.pkl'),format);return send(data,mime,f'leadiq-cleaned.{format}')
@router.get('/projects/{pid}/eda')
def get_eda(pid:str,p=Depends(auth)):require(pid,2);return plain(eda(pid))
@router.post('/projects/{pid}/features')
def build_features(pid:str,body:FeatureOptions,p=Depends(auth)):idle(pid);require(pid,2);return start_job(pid,'features',lambda log:feature_job(pid,body.pretrained,log))
@router.get('/projects/{pid}/features')
def get_features(pid:str,p=Depends(auth)):return read(pid,'features')
@router.post('/projects/{pid}/train')
def train(pid:str,body:TrainOptions,p=Depends(auth)):
    idle(pid);require(pid,4)
    if body.mode not in ['quick','thorough']:raise ValueError('Choose Quick or Thorough training.')
    return start_job(pid,'train',lambda log:train_job(pid,body.mode,log))
@router.get('/projects/{pid}/jobs/{jid}',response_model=dict)
def job(pid:str,jid:str,p=Depends(auth)):
    j=db.jobs.find_one({'id':jid,'project_id':pid},{'_id':0})
    if not j:raise HTTPException(404,'Job not found.')
    return j
@router.get('/projects/{pid}/metrics')
def metrics(pid:str,p=Depends(auth)):return current_run(pid)
@router.get('/projects/{pid}/importance')
def importance(pid:str,p=Depends(auth)):require(pid,5);return read(pid,'importance')
@router.post('/projects/{pid}/threshold')
def threshold(pid:str,body:Threshold,p=Depends(auth)):
    m=current_run(pid,body.run_id);row,data,phash=prediction_set(pid,m,body.model)
    return plain({**evaluate(data['y'],data['probability'],body.threshold),'model':body.model,'run_id':m['run_id'],'prediction_hash':phash,'saved_threshold':row['threshold'],'exploratory':True})
@router.get('/projects/{pid}/verification')
def verification(pid:str,model:str|None=None,run_id:str|None=None,p=Depends(auth)):
    m=current_run(pid,run_id)
    if model:prediction_set(pid,m,model)
    return plain({**verify(pid,model),'run_id':m['run_id'],'model':model,'operating_points':{r['name']:r['threshold'] for r in m['models'] if not model or r['name']==model},'evaluation_rows_hash':m['evaluation_rows_hash']})
@router.get('/projects/{pid}/evaluation')
def evaluation_view(pid:str,model:str,run_id:str|None=None,p=Depends(auth)):return inspect_model(pid,model,run_id)
@router.post('/projects/{pid}/evaluation-analysis')
def evaluation_analysis(pid:str,body:InspectRequest,p=Depends(auth)):return ensure_analysis(pid,body.model,body.run_id)
@router.post('/projects/{pid}/score')
def score(pid:str,body:ScoreOptions,p=Depends(auth)):
    idle(pid);require(pid,5)
    if body.scope not in ['test','all']:raise ValueError('Choose test or all rows.')
    return start_job(pid,'score',lambda log:score_job(pid,body.scope,log))
@router.post('/projects/{pid}/score-new')
async def score_new(pid:str,file:UploadFile=File(...),sheet:str|None=Form(default=None),p=Depends(auth)):
    idle(pid);require(pid,5);content=await file.read(10*1024*1024+1);df,meta=parse_file(content,file.filename or '',sheet)
    if len(meta['sheets'])>1 and not sheet:return {'needs_sheet':True,'sheets':meta['sheets']}
    required=read(pid,'features')['required_columns'];missing=[c for c in required if c not in df]
    if missing:raise ValueError('Missing required columns: '+', '.join(missing))
    return start_job(pid,'score',lambda log:score_job(pid,'new',log,df,meta))
@router.get('/projects/{pid}/score-summary')
def score_summary(pid:str,p=Depends(auth)):require(pid,7);return read(pid,'score_summary')
@router.get('/projects/{pid}/leads')
def get_leads(pid:str,search:str='',tier:str='',page:int=Query(1,ge=1),page_size:int=Query(20,ge=1,le=100),sort:str='contact_order',descending:bool=False,filter_column:str='',filter_value:str='',top_n:int|None=Query(None,ge=1,le=100000),p=Depends(auth)):
    require(pid,7);return plain(leads(pid,search,tier,page,page_size,sort,descending,filter_column,filter_value,top_n))
@router.get('/projects/{pid}/explain/{lead_id}')
def explain(pid:str,lead_id:str,p=Depends(auth)):
    require(pid,7)
    try:result=plain(explain_lead(pid,lead_id));complete(pid,8);return result
    except ValueError:raise
    except Exception as e:raise HTTPException(422,f'Explanation unavailable: {str(e)[:250]}. Predictions remain valid; no explanation was fabricated.')
@router.get('/projects/{pid}/summary')
def summary(pid:str,p=Depends(auth)):require(pid,7);result=methodology(pid);complete(pid,9);return result
@router.get('/projects/{pid}/export')
def export(pid:str,format:str='csv',top_n:int=Query(25,ge=1,le=1000),p=Depends(auth)):
    require(pid,5 if format=='metrics' else 7)
    if format in ['csv','xlsx']:
        df=pd.read_pickle(folder(pid)/'scores.pkl').drop(columns=['row_position']);data,mime=tabular(df,format)
    elif format=='pdf':data,mime=pdf_export(pid,top_n)
    elif format=='metrics':return send(json.dumps(read(pid,'metrics'),indent=2).encode(),'application/json','metrics.json')
    elif format=='reproducibility':return send(json.dumps(methodology(pid),indent=2).encode(),'application/json','reproducibility.json')
    else:raise ValueError('Choose CSV, Excel, PDF, metrics, or reproducibility.')
    return send(data,mime,f'leadiq-ranked-leads.{format}')
@router.get('/sample.csv')
def sample_csv():data,mime=tabular(sample_data(),'csv');return send(data,mime,'synthetic-demo-leads.csv')
@router.get('/template.csv')
def template():return send(b'Business Name,Category,City,Rating,Review Count,Website,Phone,Email,Converted\n','text/csv','leadiq-template.csv')

@router.post('/projects/{pid}/download-link')
def download_link(pid:str,body:DownloadRequest,p=Depends(auth)):
    """Issue a short-lived, file-specific capability for native browser download/preview."""
    if body.kind=='cleaned':
        result=cleaned_download(pid,body.format,p)
    elif body.kind=='export':
        result=export(pid,body.format,body.top_n,p)
    else:raise ValueError('Unknown download type.')
    if body.inline and result.media_type!='application/pdf':raise ValueError('Browser preview is available for PDF reports only.')
    content=result.body
    if result.media_type=='application/pdf' and (not content.startswith(b'%PDF-') or not content.rstrip().endswith(b'%%EOF')):
        raise HTTPException(500,'The PDF report could not be completed. No incomplete file was downloaded; please retry.')
    ticket=secrets.token_urlsafe(32);digest=hashlib.sha256(ticket.encode()).hexdigest()
    directory=folder(pid)/'downloads';directory.mkdir(exist_ok=True)
    filename=result.headers['content-disposition'].split('filename="',1)[1].rstrip('"')
    stored=digest+'.bin';(directory/stored).write_bytes(content)
    expires=min(datetime.now(timezone.utc)+timedelta(minutes=5),datetime.fromisoformat(p['expires_at']))
    db.downloads.insert_one({'token_hash':digest,'project_id':pid,'stored':stored,'filename':filename,'media_type':result.media_type,'inline':body.inline,'expires_at':expires.isoformat()})
    return {'download_path':f'/downloads/{ticket}','filename':filename,'bytes':len(content),'expires_at':expires.isoformat()}

@router.get('/downloads/{ticket}')
def native_download(ticket:str):
    if not re.fullmatch(r'[A-Za-z0-9_-]{43}',ticket):raise HTTPException(404,'Download link not found.')
    record=db.downloads.find_one({'token_hash':hashlib.sha256(ticket.encode()).hexdigest()},{'_id':0})
    if not record:raise HTTPException(404,'Download link not found.')
    project_record=db.projects.find_one({'id':record['project_id']},{'_id':0,'expires_at':1})
    now=datetime.now(timezone.utc)
    if not project_record or datetime.fromisoformat(record['expires_at'])<=now or datetime.fromisoformat(project_record['expires_at'])<=now:
        raise HTTPException(410,'This download link expired. Return to your workspace and download the report again.')
    artifact=folder(record['project_id'])/'downloads'/record['stored']
    if not artifact.is_file():raise HTTPException(404,'This exported file is no longer available. Please download it again.')
    return FileResponse(artifact,media_type=record['media_type'],filename=record['filename'],content_disposition_type='inline' if record['inline'] else 'attachment',headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'})

@router.post('/cron/cleanup')
async def scheduled_cleanup(request:Request,background:BackgroundTasks,authorization:str=Header(default=''),x_webhook_id:str=Header(default='')):
    # Cron endpoints must ack 2xx immediately; enqueue/background the actual work.
    expected='Bearer '+os.environ['WEBHOOK_CRON_SECRET']
    if not secrets.compare_digest(authorization,expected):raise HTTPException(401,'Unauthorized')
    try:body=await request.json()
    except Exception:raise HTTPException(400,'Invalid JSON envelope.')
    if not isinstance(body,dict) or body.get('event')!='schedule.triggered' or not (x_webhook_id or body.get('run_id')):raise HTTPException(400,'Invalid schedule envelope.')
    run_id=x_webhook_id or body['run_id'];res=db.cron_runs.update_one({'run_id':run_id},{'$setOnInsert':{'created_at':datetime.now(timezone.utc)}},upsert=True)
    if res.upserted_id:background.add_task(cleanup)
    return {'accepted':True,'duplicate':not bool(res.upserted_id)}

app.include_router(router)