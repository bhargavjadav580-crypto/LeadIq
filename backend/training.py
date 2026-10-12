import time,warnings,uuid,hashlib,json
from datetime import datetime,timezone
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV,cross_validate,cross_val_predict,learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from storage import folder,read,save,complete,update
from evaluation import evaluate,curves,bootstrap,reliability,gain,versions
from explanations import global_importance

def train_job(pid,mode,log):
    with threadpool_limits(limits=2):_train(pid,mode,log)

def _train(pid,mode,log):
    b=joblib.load(folder(pid)/'bundle.joblib');X=b['X'];y=b['y'];tr=b['train_ids'];te=b['test_ids'];Xt=X.loc[tr];yt=y.loc[tr];Xe=X.loc[te];ye=y.loc[te]
    folds=5 if mode=='thorough' else 3;iterations=25 if mode=='thorough' else 8
    if yt.value_counts().min()<folds*2:raise ValueError(f'{mode.title()} mode needs at least {folds*2} training examples in both classes for nested calibration. Add more data or choose Quick mode.')
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42);ratio=float((1-yt).sum()/yt.sum())
    candidates=[('Logistic Regression',LogisticRegression(class_weight='balanced',max_iter=1200,random_state=42),{'model__C':[.001,.01,.05,.1,.5,1,5,10,50,100]}),('Decision Tree',DecisionTreeClassifier(class_weight='balanced',random_state=42),{'model__max_depth':[2,3,4,6,8,12,None],'model__min_samples_leaf':[3,8,15,30],'model__criterion':['gini','entropy']}),('Random Forest',RandomForestClassifier(class_weight='balanced_subsample',n_jobs=2,random_state=42),{'model__n_estimators':[80,140],'model__max_depth':[4,8,14,None],'model__min_samples_leaf':[2,5,12],'model__max_features':['sqrt',.6]}),('XGBoost',XGBClassifier(scale_pos_weight=ratio,random_state=42,n_jobs=2,tree_method='hist',eval_metric='logloss'),{'model__n_estimators':[80,140],'model__max_depth':[2,3,5],'model__learning_rate':[.03,.08,.15],'model__subsample':[.75,1.],'model__colsample_bytree':[.7,1.]}),('Dummy baseline',DummyClassifier(strategy='prior',random_state=42),{})]
    fitted={};summaries=[];total_start=time.perf_counter()
    for idx,(name,est,space) in enumerate(candidates):
        log(5+idx*11,f'{name}: fitting training-only {folds}-fold cross-validation ({iterations if space else 1} candidates)');start=time.perf_counter();pipe=Pipeline([('features',clone(b['preprocessor'])),('model',est)])
        if space:
            search=RandomizedSearchCV(pipe,space,n_iter=iterations,cv=cv,scoring={'ap':'average_precision','roc':'roc_auc'},refit='ap',n_jobs=1,random_state=42,error_score='raise',return_train_score=False)
            search.fit(Xt,yt);best=search.best_estimator_;ix=search.best_index_;result=search.cv_results_;ap=float(result['mean_test_ap'][ix]);roc=float(result['mean_test_roc'][ix]);std=float(result['std_test_roc'][ix]);params={k.replace('model__',''):v for k,v in search.best_params_.items()};n_candidates=len(result['params'])
        else:
            res=cross_validate(pipe,Xt,yt,cv=cv,scoring={'ap':'average_precision','roc':'roc_auc'},n_jobs=1,error_score='raise');ap=float(res['test_ap'].mean());roc=float(res['test_roc'].mean());std=float(res['test_roc'].std());best=pipe.fit(Xt,yt);params={'strategy':'prior'};n_candidates=1
        fitted[name]=best;summaries.append({'name':name,'cv_pr_auc':ap,'cv_roc_auc':roc,'cv_roc_std':std,'training_seconds':time.perf_counter()-start,'parameters':params,'candidates':n_candidates});log(12+idx*11,f'{name} complete: cross-validated PR-AUC {ap:.4f}')
    best_name=max(summaries,key=lambda r:r['cv_pr_auc'])['name'];best=fitted[best_name];method='isotonic' if yt.sum()>=200 else 'sigmoid'
    log(63,f'Selected {best_name} using training CV PR-AUC. Calibrating with {method}.')
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=42);cal=CalibratedClassifierCV(clone(best),method=method,cv=inner,n_jobs=1)
    oof=cross_val_predict(cal,Xt,yt,cv=cv,method='predict_proba',n_jobs=1)[:,1];pr,rec,thresholds=precision_recall_curve(yt,oof);f1=2*pr[:-1]*rec[:-1]/np.maximum(pr[:-1]+rec[:-1],1e-12);threshold=float(thresholds[np.argmax(f1)])
    log(69,f'OOF calibrated threshold selected at {threshold:.4f}; fitting final calibrated model')
    cal.fit(Xt,yt);train_ref=np.sort(cal.predict_proba(Xt)[:,1]);pcal=cal.predict_proba(Xe)[:,1]
    metrics={'run_id':uuid.uuid4().hex,'trained_at':datetime.now(timezone.utc).isoformat(),'dataset_name':read(pid,'upload')['filename'],'target':b['config']['target'],'positive_class':b['config']['positive'],'evaluation_rows_hash':hashlib.sha256(json.dumps(te).encode()).hexdigest(),'best_model':best_name,'mode':mode,'folds':folds,'iterations':iterations,'calibration_method':method,'threshold':threshold,'seed':42,'prevalence':float(ye.mean()),'split':read(pid,'features')['split'],'models':[],'warnings':list(read(pid,'features')['warnings']),'versions':versions(),'dataset_hash':read(pid,'upload')['hash'],'test_rows':te,'selection_metric':'training cross-validated average precision','probability_reference':'calibrated probabilities on training records'}
    log(73,'Evaluating frozen candidates once on the untouched test set; saving every prediction')
    for i,row in enumerate(summaries):
        name=row['name'];uncal=fitted[name].predict_proba(Xe)[:,1];p=pcal if name==best_name else uncal;t=threshold if name==best_name else .5
        row.update(evaluate(ye,p,t));row.update(curves(ye,p));row['calibrated']=name==best_name;row['prediction_file']=f'predictions_{i}.npz';np.savez(folder(pid)/row['prediction_file'],y=ye.to_numpy(),probability=p,uncalibrated=uncal,threshold=t);metrics['models'].append(row)
    winner=next(r for r in metrics['models'] if r['name']==best_name);metrics['best']=winner
    ci,skipped=bootstrap(ye,pcal,threshold,log);metrics['confidence_intervals']=ci;metrics['bootstrap']={'resamples':1000,'undefined_resamples_skipped':skipped,'seed':42}
    metrics['calibration']={'before':reliability(ye,best.predict_proba(Xe)[:,1]),'after':reliability(ye,pcal)};metrics['gain']=gain(ye,pcal)
    baseline=next(r for r in metrics['models'] if r['name']=='Dummy baseline');metrics['baseline_improvement_pp']=100*(winner['pr_auc']-baseline['pr_auc'])
    if winner['roc_auc']>.97:metrics['warnings'].append('Possible data leakage: test ROC-AUC exceeds 0.97. Review '+(', '.join(v['column'] for v in b['config'].get('leakage',[])) or ', '.join(b['spec']['required'][:8]))+'. High performance alone does not prove leakage.')
    if winner['roc_auc']<.55:metrics['warnings'].append('The model found little signal in this data (test ROC-AUC below 0.55).')
    log(87,'Measuring learning curves at 20%, 40%, 60%, 80%, and 100% of training data')
    metrics['learning_curve']=[]
    try:
        sizes,train_scores,val_scores=learning_curve(clone(best),Xt,yt,train_sizes=[.2,.4,.6,.8,1.],cv=cv,scoring='average_precision',shuffle=True,random_state=42,n_jobs=1,error_score=np.nan)
        for i,size in enumerate(sizes):
            valid=np.isfinite(val_scores[i]);metrics['learning_curve'].append({'fraction':[.2,.4,.6,.8,1.][i],'size':int(size),'train':float(np.nanmean(train_scores[i])) if valid.any() else None,'validation':float(np.nanmean(val_scores[i])) if valid.any() else None,'std':float(np.nanstd(val_scores[i])) if valid.any() else None,'available':bool(valid.any())})
    except Exception as e:metrics['learning_notice']=f'Learning curve unavailable: {str(e)[:250]}'
    importance=global_importance(pid,best,Xt,Xe,ye,log)
    if importance.get('status')=='unavailable':metrics['warnings'].append(importance['notice'])
    joblib.dump({'best_pipeline':best,'calibrated':cal,'models':fitted,'train_reference':train_ref,'threshold':threshold,'bundle':b,'best_name':best_name},folder(pid)/'trained.joblib')
    metrics['total_seconds']=time.perf_counter()-total_start;save(pid,'metrics',metrics);update(pid,active_run_id=metrics['run_id']);complete(pid,5);log(100,'All results saved. Predictions, confidence intervals, and model evidence are ready.')

def verify(pid,model=None):
    metrics=read(pid,'metrics');rows=[]
    for m in metrics['models']:
        if model and m['name']!=model:continue
        p=np.load(folder(pid)/m['prediction_file']);fresh=evaluate(p['y'],p['probability'],float(p['threshold']))
        for key in ['accuracy','precision','recall','f1','roc_auc','pr_auc','precision_top10','confusion_matrix']:
            rows.append({'model':m['name'],'metric':key,'displayed':m[key],'recomputed':fresh[key],'match':bool(np.allclose(np.asarray(m[key]),np.asarray(fresh[key]),atol=1e-10))})
    if not rows:raise ValueError('The requested model does not belong to this completed run.')
    result={'rows':rows,'all_match':all(v['match'] for v in rows),'seed':42,'versions':metrics['versions'],'split':metrics['split'],'dataset_hash':metrics['dataset_hash']};complete(pid,6);return result