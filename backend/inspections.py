"""Model-specific inspection of immutable saved predictions, never model selection."""
import hashlib,json,time,warnings
import numpy as np
import joblib
from fastapi import HTTPException
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold,learning_curve
from threadpoolctl import threadpool_limits
from storage import db,folder,read,save,start_job,plain
from evaluation import evaluate,gain,reliability
from explanations import global_importance

def current_run(pid,expected=None):
    p=db.projects.find_one({'id':pid},{'_id':0,'completed':1})
    if not p or 5 not in p.get('completed',[]):
        raise HTTPException(409,'Evaluation is unavailable for the current configuration. Complete a successful training run in Step 5; previous results have been invalidated.')
    m=read(pid,'metrics')
    # Existing completed projects remain inspectable without retraining or changing their numbers.
    m.setdefault('run_id','retained-'+hashlib.sha256((folder(pid)/'metrics.json').read_bytes()).hexdigest()[:20])
    if expected and expected!=m['run_id']:raise HTTPException(409,'The training run changed. Refresh this view to inspect the current run.')
    cfg=read(pid,'config');upload=read(pid,'upload')
    m.setdefault('target',cfg['target']);m.setdefault('positive_class',cfg['positive']);m.setdefault('dataset_name',upload['filename'])
    m.setdefault('evaluation_rows_hash',hashlib.sha256(json.dumps(m['test_rows']).encode()).hexdigest())
    return m

def prediction_set(pid,m,model):
    row=next((r for r in m['models'] if r['name']==model),None)
    if row is None:raise HTTPException(404,'This model is not part of the current training run.')
    file=folder(pid)/row['prediction_file']
    with np.load(file) as stored:data={k:stored[k].copy() for k in stored.files}
    return row,data,hashlib.sha256(file.read_bytes()).hexdigest()

def cache_name(run_id,model):
    return 'inspection_'+hashlib.sha256(f'{run_id}:{model}'.encode()).hexdigest()

def inspect_model(pid,model,expected=None):
    m=current_run(pid,expected);row,p,phash=prediction_set(pid,m,model);winner=model==m['best_model'];cached=None
    artifact=cache_name(m['run_id'],model)
    if (folder(pid)/(artifact+'.json')).exists():cached=read(pid,artifact)
    importance=read(pid,'importance') if winner and (folder(pid)/'importance.json').exists() else None
    lc=m.get('learning_curve') if winner else None;lc_notice=m.get('learning_notice') if winner else None
    if cached:
        if cached['run_id']!=m['run_id'] or cached['model']!=model:raise HTTPException(409,'Analysis cache does not match this run.')
        importance=cached['importance'];lc=cached['learning_curve'];lc_notice=cached.get('learning_notice')
    values=evaluate(p['y'],p['probability'],float(p['threshold']))
    calibration={'calibrated':winner,'method':m['calibration_method'] if winner else None,'current':reliability(p['y'],p['probability'])}
    if winner:
        calibration.update(before=reliability(p['y'],p['uncalibrated']),after=calibration['current'],notice='Before/after reliability for the scoring winner on the same held-out records.')
    else:calibration['notice']='This candidate was not calibrated in this run. Only its actual uncalibrated reliability is available; no after-calibration curve exists.'
    local_warnings=[]
    if values['roc_auc'] is not None and values['roc_auc']>.97:local_warnings.append('Possible leakage: this model’s test ROC-AUC exceeds 0.97. Review outcome-like and suspicious input columns; high performance alone is not proof of leakage.')
    if values['roc_auc'] is not None and values['roc_auc']<.55:local_warnings.append('This model found little signal in the evaluated records (ROC-AUC below 0.55).')
    context={'dataset':m['dataset_name'],'dataset_hash':m['dataset_hash'],'target':m['target'],'positive_class':m['positive_class'],'run_id':m['run_id'],'trained_at':m.get('trained_at'),'evaluation_rows_hash':m['evaluation_rows_hash'],'prediction_hash':phash,'split':m['split'],'test_positives':int(p['y'].sum()),'test_negatives':int(len(p['y'])-p['y'].sum()),'test_rows':len(p['y']),'prevalence':float(p['y'].mean()),'saved_threshold':float(p['threshold']),'selection_criterion':m['selection_metric']}
    return plain({'run_id':m['run_id'],'model':model,'scoring_model':m['best_model'],'metrics':{**row,'undefined':values['undefined']},'context':context,'gain':gain(p['y'],p['probability']),'calibration':calibration,'confidence_intervals':m['confidence_intervals'] if winner else None,'uncertainty_notice':'95% bootstrap intervals apply only to this winner’s saved predictions and operating threshold (1,000 resamples).' if winner else 'Confidence intervals were not computed for this candidate. Winner intervals are not substituted.','learning_curve':lc,'learning_notice':lc_notice,'learning_scope':'Uncalibrated candidate, cloned and fitted within training-only cross-validation. Fractions refer to each CV training fold.','importance':importance,'analysis_status':'ready' if (lc is not None or lc_notice) and importance is not None else 'not_calculated','warnings':local_warnings})

def calculate_model_analysis(pid,model,run_id,log):
    m=current_run(pid,run_id);start=time.perf_counter();t=joblib.load(folder(pid)/'trained.joblib');b=t['bundle'];pipe=t['models'].get(model)
    if pipe is None:raise ValueError('The fitted pipeline for this model is unavailable in the saved run.')
    X=b['X'];y=b['y'];Xt=X.loc[b['train_ids']];yt=y.loc[b['train_ids']];Xe=X.loc[b['test_ids']];ye=y.loc[b['test_ids']]
    result={'run_id':m['run_id'],'model':model,'learning_curve':[],'learning_notice':None}
    with threadpool_limits(limits=2):
        log(5,f'{model}: calculating model-specific training-only learning curves')
        cv=StratifiedKFold(n_splits=m['folds'],shuffle=True,random_state=42)
        try:
            with warnings.catch_warnings(record=True) as caught:
                sizes,train,val=learning_curve(clone(pipe),Xt,yt,train_sizes=[.2,.4,.6,.8,1.],cv=cv,scoring='average_precision',shuffle=True,random_state=42,n_jobs=1,error_score=np.nan)
            for i,size in enumerate(sizes):
                valid=np.isfinite(val[i]);training_valid=np.isfinite(train[i]);available=bool(valid.any())
                result['learning_curve'].append({'fraction':[.2,.4,.6,.8,1.][i],'size':int(size),'train':float(np.mean(train[i][training_valid])) if training_valid.any() else None,'validation':float(np.mean(val[i][valid])) if available else None,'std':float(np.std(val[i][valid])) if available else None,'available':available,'valid_folds':int(valid.sum()),'total_folds':m['folds'],'reason':None if valid.all() else 'Some folds could not be fitted or evaluated at this training volume; only finite fold scores are summarized.'})
            if caught:result['learning_notice']='One or more folds raised fitting or metric warnings. Check the valid-fold counts; no unavailable point has been substituted.'
        except Exception as e:result['learning_notice']=f'Learning curve unavailable for {model}: {str(e)[:250]}'
        log(60,f'{model}: learning-curve calculation finished; computing actual feature influence')
        result['importance']=global_importance(pid,pipe,Xt,Xe,ye,lambda _,text:log(80,text),artifact_name=None,save_background=False)
    current_run(pid,run_id)
    result['seconds']=time.perf_counter()-start;save(pid,cache_name(run_id,model),result);log(100,f'{model}: model-specific analysis saved for this run')

def ensure_analysis(pid,model,run_id):
    state=inspect_model(pid,model,run_id)
    if state['analysis_status']=='ready':return {'result':state}
    active=db.jobs.find_one({'project_id':pid,'kind':'inspection','status':{'$in':['queued','running']}},{'_id':0,'id':1})
    if active:return {'job_id':active['id']}
    job=start_job(pid,'inspection',lambda log:calculate_model_analysis(pid,model,state['run_id'],log))
    db.jobs.update_one({'id':job['job_id']},{'$set':{'model':model,'run_id':state['run_id']}})
    return job