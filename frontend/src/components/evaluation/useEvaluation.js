import {useState,useEffect,useRef,useCallback} from 'react';
import {api,path,message} from '../../lib/api';

export function useEvaluation(){
 const [run,setRun]=useState(null),[model,setModel]=useState(''),[inspection,setInspection]=useState(null),[error,setError]=useState(''),[loading,setLoading]=useState(true),[analysisJob,setAnalysisJob]=useState(null),[analysisError,setAnalysisError]=useState('');
 const current=useRef({model:'',run:''});
 useEffect(()=>{const c=new AbortController();api.get(path('/metrics'),{signal:c.signal}).then(r=>{setRun(r.data);setModel(r.data.best_model)}).catch(e=>{if(!c.signal.aborted){setError(message(e));setLoading(false)}});return()=>c.abort()},[]);
 useEffect(()=>{if(!run||!model)return;const c=new AbortController();current.current={model,run:run.run_id};setInspection(null);setLoading(true);setError('');setAnalysisJob(null);setAnalysisError('');api.get(path('/evaluation'),{params:{model,run_id:run.run_id},signal:c.signal}).then(r=>setInspection(r.data)).catch(e=>{if(!c.signal.aborted)setError(message(e))}).finally(()=>{if(!c.signal.aborted)setLoading(false)});return()=>c.abort()},[model,run]);
 const calculate=useCallback(async()=>{if(!run||!model)return;const requested={model,run:run.run_id};setAnalysisError('');try{const r=await api.post(path('/evaluation-analysis'),{model,run_id:run.run_id});if(current.current.model!==requested.model||current.current.run!==requested.run)return;if(r.data.result)setInspection(r.data.result);else setAnalysisJob(r.data.job_id)}catch(e){if(current.current.model===requested.model&&current.current.run===requested.run)setAnalysisError(message(e))}},[run,model]);
 useEffect(()=>{if(!analysisJob)return;let live=true,timer;const requested={...current.current};const c=new AbortController();const poll=async()=>{try{const r=await api.get(path(`/jobs/${analysisJob}`),{signal:c.signal});if(!live)return;if(r.data.status==='failed'){setAnalysisError(r.data.error||'Analysis failed. Retry when ready.');setAnalysisJob(null);return}if(r.data.status==='completed'){const next=await api.get(path('/evaluation'),{params:{model:requested.model,run_id:requested.run},signal:c.signal});if(live&&current.current.model===requested.model&&current.current.run===requested.run){setInspection(next.data);setAnalysisJob(null)}return}timer=setTimeout(poll,1300)}catch(e){if(live){setAnalysisError(message(e));setAnalysisJob(null)}}};poll();return()=>{live=false;c.abort();clearTimeout(timer)}},[analysisJob]);
 return {run,model,setModel,inspection,error,loading,analysisJob,analysisError,calculate};
}

export function useThreshold(run,model,savedThreshold){
 const [threshold,setThreshold]=useState(savedThreshold??0.5),[result,setResult]=useState(null),[busy,setBusy]=useState(true),[error,setError]=useState('');const sequence=useRef(0);
 useEffect(()=>{setThreshold(savedThreshold??0.5);setResult(null)},[run,model,savedThreshold]);
 useEffect(()=>{
  if(!run||!model||savedThreshold==null)return;
  const id=++sequence.current,c=new AbortController();
  setBusy(true);setResult(null);setError('');
  const timer=setTimeout(()=>{
   api.post(path('/threshold'),{model,threshold,run_id:run},{signal:c.signal})
    .then(r=>{if(id===sequence.current&&!c.signal.aborted)setResult(r.data)})
    .catch(e=>{if(id===sequence.current&&!c.signal.aborted)setError(message(e))})
    .finally(()=>{if(id===sequence.current&&!c.signal.aborted)setBusy(false)});
  },180);
  return()=>{c.abort();clearTimeout(timer)};
 },[run,model,threshold,savedThreshold]);
 const change=value=>{sequence.current+=1;setResult(null);setBusy(true);setThreshold(value)};
 return {threshold,setThreshold:change,result,busy,error};
}