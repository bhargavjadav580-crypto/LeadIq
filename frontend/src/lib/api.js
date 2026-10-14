import axios from 'axios';
export const api = axios.create({baseURL: `${process.env.REACT_APP_BACKEND_URL}/api`});
export const credentials = () => {try{return JSON.parse(localStorage.getItem('leadiq-project') || 'null')}catch{return null}};
api.interceptors.request.use(config => {const p=credentials();if(p)config.headers['X-Project-Key']=p.token;return config});
export const path = (suffix='') => `/projects/${credentials()?.id}${suffix}`;
export const message = e => typeof e.response?.data?.detail === 'string' ? e.response.data.detail : e.message || 'Something went wrong. Please try again.';
async function downloadLink(suffix,inline=false){
 const query=new URLSearchParams(suffix.split('?')[1]||'');
 const r=await api.post(path('/download-link'),{kind:suffix.startsWith('/cleaned')?'cleaned':'export',format:query.get('format')||'csv',export_kind:query.get('kind')||'ranked',top_n:Number(query.get('top_n')||25),inline});
 return {url:`${api.defaults.baseURL}${r.data.download_path}`,filename:r.data.filename};
}
export async function download(suffix,name,publicFile=false){
 const file=publicFile?{url:`${api.defaults.baseURL}${suffix}`,filename:name}:await downloadLink(suffix);
 // A real HTTPS attachment avoids revoking a blob before Chrome has finished saving it.
 const a=document.createElement('a');a.href=file.url;a.download=file.filename;a.rel='noopener';a.style.display='none';
 document.body.appendChild(a);a.click();setTimeout(()=>a.remove(),60000);
}
export async function openPdf(suffix){
 // Open synchronously from the user's click so popup blockers can recognize the gesture.
 const tab=window.open('about:blank','_blank');
 if(!tab)throw new Error('Your browser blocked the PDF tab. Allow pop-ups for this workspace, or use Download PDF.');
 tab.opener=null;tab.document.title='Preparing your LeadIQ report';tab.document.body.textContent='Preparing your PDF report…';
 try{const file=await downloadLink(suffix,true);tab.location.replace(file.url)}catch(e){tab.close();throw e}
}