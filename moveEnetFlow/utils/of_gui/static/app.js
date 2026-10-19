'use strict';
const $ = id => document.getElementById(id);
const jointNames = ['Head','Right shoulder','Left shoulder','Right elbow','Left elbow','Left hip','Right hip','Right wrist','Left wrist','Right knee','Left knee','Right ankle','Left ankle'];
const bones = [[0,1],[0,2],[1,2],[1,3],[3,7],[2,4],[4,8],[1,6],[2,5],[5,6],[5,10],[10,12],[6,9],[9,11]];
let settingsLoaded = false, paused = false, duration = 0, currentTime = 0, activeGeneration = -1;
let currentFrame = null, images = null, frameSeq = -1, dragged = false, applyTimer = null;
let selectedFile = '', folder = '/data', chartHistory = [], recording = false, mediaRecorder = null, captured = [];
const mainCanvas = $('recordCanvas');

function notify(message, error=false) {
  const toast=$('toast'); toast.textContent=message;toast.hidden=false;
  toast.style.borderColor=error?'#ee7a7a':'#70bdcb';
  clearTimeout(notify.timer);notify.timer=setTimeout(()=>toast.hidden=true,4500);
}
async function request(path, method='GET', body=null) {
  const res=await fetch(path, {method, headers:body===null?{}:{'Content-Type':'application/json'},
    body:body===null?null:JSON.stringify(body), cache:'no-store'});
  const data=await res.json();
  if(!res.ok)throw new Error(data.error||res.statusText);
  return data;
}
async function safeRequest(path, method, body) {
  try{return await request(path,method,body);}catch(e){notify(e.message,true);return null;}
}
function formSettings() {
  const path=$('samplePath').value.trim();
  const d=$('dataset').value;
  const isDhp = d==='dhp19'||(d==='auto'&&path.toLowerCase().includes('dhp19'));
  return {
    data_file:path, dhp19:isDhp, w:+$('sensorW').value, h:+$('sensorH').value,
    net_period:+$('netPeriod').value, flow_period:+$('flowPeriod').value,
    fps:+$('fps').value, gain_a:+$('gainA').value,gain_b:+$('gainB').value,
    compare:$('compare').checked,roi:+$('roiSize').value,
    of_joints:Array.from(document.querySelectorAll('.ofJoint:checked')).map(e=>+e.dataset.joint),
    reset_on_detection:$('integrationMode').value==='reset',
    pu:+$('pu').value,muD:+$('muD').value,muV:+$('muV').value,use_lc:$('useLC').checked,
    device:$('device').value.trim(),event_window_ms:30,
    movenet_script:$('scriptPath').value.trim(),
    checkpoint_path:$('checkpointPath').value.trim(), python:'python3'
  };
}
function fillForm(s) {
  const vals={samplePath:s.data_file, dataset:s.dhp19?'dhp19':'eh36m',sensorW:s.w,sensorH:s.h,
    netPeriod:s.net_period,flowPeriod:s.flow_period,fps:s.fps,
    gainA:s.gain_a,gainANum:s.gain_a,gainB:s.gain_b,gainBNum:s.gain_b,
    roiSize:s.roi,pu:s.pu,muD:s.muD,muV:s.muV,device:s.device,
    integrationMode:s.reset_on_detection?'reset':'continuous',
    scriptPath:s.movenet_script,checkpointPath:s.checkpoint_path};
  for(const [key,value] of Object.entries(vals)) if($(key))$(key).value=value;
  $('compare').checked=s.compare;$('useLC').checked=s.use_lc;
  document.querySelectorAll('.ofJoint').forEach(x=>x.checked=s.of_joints.includes(+x.dataset.joint));
  $('gainATag').textContent=(+s.gain_a).toFixed(2);
  $('gainBTag').textContent=(+s.gain_b).toFixed(2);
  $('gainBGroup').hidden=!s.compare;
  updateSelectedFilePath();
}
function makeJoints() {
  for(let i=0;i<13;i++) {
    const line=document.createElement('div'); line.className='jointrow';
    const name=document.createElement('span');name.textContent=`${String(i).padStart(2,'0')} · ${jointNames[i]}`;
    const show=document.createElement('input');show.type='checkbox';show.checked=true;show.className='displayJoint';show.dataset.joint=i;
    const active=document.createElement('input');active.type='checkbox';active.checked=true;active.className='ofJoint algorithm';active.dataset.joint=i;
    line.append(name,show,active);$('jointList').appendChild(line);
    let opt=document.createElement('option');opt.value=i;opt.textContent=jointNames[i];$('selectedJoint').appendChild(opt);
    show.addEventListener('change',render);
    active.addEventListener('change',scheduleRecompute);
  }
  $('selectedJoint').value='7';
  $('selectedJoint').addEventListener('change',()=>{chartHistory=[];render();});
}
function displayed() {
  return new Set(Array.from(document.querySelectorAll('.displayJoint:checked')).map(e=>+e.dataset.joint));
}
function setCheckList(selector,state) {
  document.querySelectorAll(selector).forEach(e=>e.checked=state);
  if(selector==='.ofJoint')scheduleRecompute(); else render();
}
$('displayAll').onclick=()=>setCheckList('.displayJoint',true);
$('displayNone').onclick=()=>setCheckList('.displayJoint',false);
$('computeAll').onclick=()=>setCheckList('.ofJoint',true);
$('computeNone').onclick=()=>setCheckList('.ofJoint',false);

// Full-sized server-side file browser. Paths are never truncated or rewritten.
let browserEntries = [], browserSearching = false, browserRequestSeq = 0;
let browserSelected = '';
const escapeLabel = s => String(s ?? '');
function browserStatus(message) { $('browseStatus').textContent = message; }
function openBrowser() {
  const current = $('samplePath').value.trim();
  const initial = current.startsWith('/') ? (current.endsWith('/data.log') ? current.slice(0,-9) : current) : folder;
  $('fileDialog').hidden=false;
  document.body.classList.add('dialog-open');
  $('fileFilter').value='';$('fileSearch').value='';
  browserSelected='';updateSelectedPath();
  browse(initial || '/data');
  $('folderInput').focus();
}
function closeBrowser() {
  $('fileDialog').hidden=true;
  document.body.classList.remove('dialog-open');
  ++browserRequestSeq; // Ignore late file responses when modal is closed.
}
function updateSelectedPath(){
  $('selectedPath').textContent=browserSelected || 'No file selected';
  $('selectedPath').title=browserSelected;
  $('chooseSelected').disabled=!browserSelected;
}
function chooseFile(path) {
  if(!path.endsWith('/data.log')){browserStatus('Please choose a data.log file.');return;}
  browserSelected=path;
  $('samplePath').value=path;
  selectedFile=path;
  updateDatasetDefaults();
  closeBrowser();
  notify('Selected: '+path);
}
function breadcrumbs(path){
  const parent=$('breadcrumbs');parent.replaceChildren();
  const parts=path.split('/').filter(Boolean);
  let full='';
  const root=document.createElement('button');root.textContent='/';root.className='crumb';root.onclick=()=>browse('/');parent.append(root);
  for(const part of parts){
    full+='/'+part;
    const pathNow=full;
    const item=document.createElement('button');item.className='crumb';item.textContent=part;item.title=pathNow;
    item.onclick=()=>browse(pathNow);
    parent.append(item);
  }
}
function renderBrowserEntries(){
  const needle=$('fileFilter').value.trim().toLowerCase();
  const entries=needle ? browserEntries.filter(e=>e.name.toLowerCase().includes(needle)||e.path.toLowerCase().includes(needle)) : browserEntries;
  const list=$('fileResults');list.replaceChildren();
  if(!entries.length){
    const msg=document.createElement('div');msg.className='browse-empty';
    msg.textContent=browserSearching?'No matching data.log found in the search scope.':'No matching entries. Open a different folder or use recursive search.';
    list.append(msg);
    return;
  }
  for(const entry of entries){
    const row=document.createElement('button');row.type='button';row.className='file-entry '+(entry.kind==='dir'?'directory':'log-file');
    if(entry.path===browserSelected)row.classList.add('selected');
    row.title=entry.path;
    const icon=document.createElement('span');icon.className='file-icon';icon.textContent=entry.kind==='dir'?'▸':'●';
    const text=document.createElement('span');text.className='file-entry-text';
    const name=document.createElement('strong');name.textContent=entry.name;
    const path=document.createElement('small');path.textContent=entry.path;
    text.append(name,path);row.append(icon,text);
    row.onclick=()=>{
      if(entry.kind==='dir') { $('fileSearch').value='';browse(entry.path); }
      else chooseFile(entry.path);
    };
    list.append(row);
  }
}
async function browse(path) {
  const seq=++browserRequestSeq;
  browserSearching=false;
  browserStatus('Loading '+path+'…');
  try{
    const info=await request('/api/list?path='+encodeURIComponent(path));
    if(seq!==browserRequestSeq || $('fileDialog').hidden)return;
    folder=info.path;
    $('folderInput').value=folder;
    breadcrumbs(folder);
    browserEntries=info.items;
    renderBrowserEntries();
    browserStatus(info.items.length+' entries'+(info.truncated?' (more exist; use folder navigation or search)':'')+'. Double-click is not required: click a directory to open or data.log to select.');
  }catch(e){if(seq!==browserRequestSeq)return;browserStatus('Error: '+e.message);notify(e.message,true);}
}
async function searchBrowser(){
  const seq=++browserRequestSeq;
  browserSearching=true;
  const needle=$('fileSearch').value.trim();
  const path=folder;
  $('fileFilter').value='';
  browserStatus('Searching recursively under '+path+'…');
  try{
    const info=await request('/api/search?path='+encodeURIComponent(path)+'&q='+encodeURIComponent(needle));
    if(seq!==browserRequestSeq || $('fileDialog').hidden)return;
    browserEntries=info.items;
    renderBrowserEntries();
    browserStatus(info.items.length+' matching data.log files found'+(info.truncated?' (search capped; narrow the folder or search term)':'')+'. Select a file to use it.');
  }catch(e){if(seq!==browserRequestSeq)return;browserStatus('Search failed: '+e.message);notify(e.message,true);}
}
async function jumpToPath(){
  const raw=$('folderInput').value.trim();
  if(!raw.startsWith('/')){browserStatus('Enter an absolute path starting with /.');return;}
  if(raw.endsWith('/data.log')){
    try{
      const info=await request('/api/inspect?path='+encodeURIComponent(raw));
      if(info.is_file){chooseFile(info.path);return;}
      browserStatus('File was not found: '+raw);
    }catch(e){browserStatus(e.message);notify(e.message,true);}
  } else { $('fileSearch').value='';$('fileFilter').value='';browse(raw); }
}
$('browse').onclick=openBrowser;
$('fileDialogClose').onclick=closeBrowser;
$('fileDialog').addEventListener('click',e=>{if(e.target===$('fileDialog'))closeBrowser();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('fileDialog').hidden)closeBrowser();});
$('goFolder').onclick=jumpToPath;
$('folderInput').addEventListener('keydown',e=>{if(e.key==='Enter')jumpToPath();});
$('parentFolder').onclick=()=>{const p=folder.replace(/\/+$/,'');browse(p.substring(0,p.lastIndexOf('/'))||'/');};
$('fileFilter').addEventListener('input',renderBrowserEntries);
$('fileSearch').addEventListener('keydown',e=>{if(e.key==='Enter')searchBrowser();});
$('searchFiles').onclick=searchBrowser;
$('chooseSelected').onclick=()=>browserSelected&&chooseFile(browserSelected);
$('toggleSidebar').onclick=()=>{
  const hidden=$('appShell').classList.toggle('sidebar-collapsed');
  $('toggleSidebar').textContent=hidden?'Show settings':'Hide settings';
};
$('samplePath').addEventListener('change',updateDatasetDefaults);
$('samplePath').addEventListener('input',updateSelectedFilePath);
function updateSelectedFilePath(){
  const path=$('samplePath').value.trim();
  $('selectedFilePath').hidden=!path;
  $('selectedFilePath').textContent=path;
  $('selectedFilePath').title=path;
}
function updateDatasetDefaults() {
  updateSelectedFilePath();
  const d=$('dataset').value, path=$('samplePath').value.toLowerCase();
  const dhp=d==='dhp19'||(d==='auto'&&path.includes('dhp19'));
  $('sensorW').value=dhp?346:640;$('sensorH').value=dhp?260:480;
  $('pu').value=dhp?0.36:0.77;$('muD').value=dhp?0.01:0.06;$('muV').value=dhp?0.01:0.97;
}
$('dataset').addEventListener('change',updateDatasetDefaults);
$('openSample').onclick=async()=>{
  const s=await safeRequest('/api/open','POST',{settings:formSettings()});
  if(s){settingsLoaded=true;paused=false;activeGeneration=s.generation;frameSeq=s.seq;currentFrame=null;images=null;chartHistory=[];updateState(s);}
};
function scheduleRecompute() {
  if(!settingsLoaded)return;
  clearTimeout(applyTimer);
  applyTimer=setTimeout(async()=>{
    const s=await safeRequest('/api/apply','POST',{settings:formSettings(),time:currentTime,paused});
    if(s){activeGeneration=s.generation; frameSeq=s.seq; chartHistory=[];updateState(s);}
  },500);
}
for(const elem of document.querySelectorAll('.algorithm'))elem.addEventListener('change',scheduleRecompute);
function bindGain(slider,num,tag){
  const range=$(slider),numeric=$(num);
  range.addEventListener('input',()=>{numeric.value=range.value;$(tag).textContent=(+range.value).toFixed(2);scheduleRecompute();});
  numeric.addEventListener('input',()=>{
    range.value=numeric.value;$(tag).textContent=(+numeric.value).toFixed(2);scheduleRecompute();
  });
}
bindGain('gainA','gainANum','gainATag');bindGain('gainB','gainBNum','gainBTag');
$('compare').addEventListener('change',()=>{$('gainBGroup').hidden=!$('compare').checked;});

async function control(action,value){const s=await safeRequest('/api/control','POST',{action,value});if(s)updateState(s);}
$('playPause').onclick=()=>control('pause',!paused);
$('speed').onchange=()=>control('speed',+$('speed').value);
$('stop').onclick=async()=>{if(!settingsLoaded)return;const s=await safeRequest('/api/restart','POST',{paused:true});if(s){activeGeneration=s.generation;frameSeq=s.seq;chartHistory=[];updateState(s);}};
$('restart').onclick=async()=>{if(!settingsLoaded)return;const s=await safeRequest('/api/restart','POST',{paused:false});if(s){activeGeneration=s.generation;frameSeq=s.seq;chartHistory=[];updateState(s);}};
$('timeline').addEventListener('input',()=>{dragged=true;$('position').textContent=formatTime(+$('timeline').value);});
$('timeline').addEventListener('change',async()=>{
  if(!settingsLoaded)return;
  const target=+$('timeline').value;dragged=false;
  const s=await safeRequest('/api/seek','POST',{time:target,paused});
  if(s){activeGeneration=s.generation;frameSeq=s.seq;chartHistory=[];updateState(s);}
});
function formatTime(t){t=Math.max(0,t||0);return `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')}`;}
function updateState(s){
  if(!s)return;
  paused=!!s.paused;duration=s.duration||duration;currentTime=s.time||0;
  if(s.generation!==activeGeneration){activeGeneration=s.generation;chartHistory=[];}
  $('serverBadge').textContent=s.error?'Engine error':s.busy?'Recalculating…':s.finished?'End of sample':s.ready?'Live playback':'Waiting for sample';
  $('serverBadge').style.borderColor=s.error?'#d66e6e':'#4a6774';
  $('infoLabel').textContent=s.file?s.file.split('/').slice(-3).join('/'): 'No recording loaded';
  $('playPause').textContent=paused?'▶':'Ⅱ';
  $('duration').textContent=formatTime(duration);
  if(!dragged){$('position').textContent=formatTime(currentTime);$('timeline').max=Math.max(duration,1);$('timeline').value=currentTime;}
  $('loader').hidden=!s.busy;
  $('progressText').textContent=s.progress>0 ? `Processed ${formatTime(s.progress)} / ${formatTime(duration)} before target`:'Initializing / scanning sample…';
  if(s.error && s.error!==lastError){lastError=s.error;notify(s.error,true);}
  if(!s.error)lastError='';
}
let lastError='';

const panels={left:$('leftCanvas'),right:$('rightCanvas')};
const optids=['showNet','showFused','showIntegrated','showCompare','showArrows','showRois','showLabels','showTime','leftBg','rightBg','layout','solo','arrowMode'];
for(let id of optids)$(id).addEventListener('change',render);
function configureLayout(){
  const layout=$('layout').value;const stage=$('videoStage');
  stage.className='video-stage '+(layout==='side'?'horiz':layout)+(layout==='single'?(' focus-'+$('solo').value):'');
  $('solo').disabled=layout!=='single';
}
function validPt(p,W,H){return p && Number.isFinite(p[0])&&Number.isFinite(p[1]) &&
  (p[0]>0||p[1]>0)&&p[0]>=0&&p[1]>=0&&p[0]<W&&p[1]<H;}
function skeleton(ctx,pose,color,sel,W,H,lineWidth=2){
  if(!pose)return;
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=lineWidth;
  for(const [a,b] of bones){if(!sel.has(a)||!sel.has(b)||!validPt(pose[a],W,H)||!validPt(pose[b],W,H))continue;
    ctx.beginPath();ctx.moveTo(...pose[a]);ctx.lineTo(...pose[b]);ctx.stroke();}
  for(let i=0;i<13;i++){
    if(!sel.has(i)||!validPt(pose[i],W,H))continue;
    ctx.beginPath();ctx.arc(pose[i][0],pose[i][1],3.2,0,2*Math.PI);ctx.fill();
    if($('showLabels').checked){ctx.font='11px sans-serif';ctx.fillText(String(i),pose[i][0]+4,pose[i][1]-5);}
  }
}
function arrow(ctx,x,y,dx,dy,color){
  const length=Math.hypot(dx,dy);if(length<.1||length>5000)return;
  const ex=x+dx,ey=y+dy,angle=Math.atan2(dy,dx),head=7;
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.stroke();
  ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-head*Math.cos(angle-.5),ey-head*Math.sin(angle-.5));
  ctx.moveTo(ex,ey);ctx.lineTo(ex-head*Math.cos(angle+.5),ey-head*Math.sin(angle+.5));ctx.stroke();
}
function drawVectors(ctx,lane,sel,mask,W,H,color){
  if(!lane)return;
  const mode=$('arrowMode').value;
  for(const i of mask){
    const p=lane.centres[i],v=lane.velocity[i];if(!sel.has(i)||!validPt(p,W,H)||!v)continue;
    let dx=v[0],dy=v[1],length=Math.hypot(dx,dy);
    if(length<1e-6)continue;
    if(mode==='normalized'){dx=dx/length*20;dy=dy/length*20;}
    else {dx*=.05;dy*=.05;}
    arrow(ctx,p[0],p[1],dx,dy,color);
  }
}
function drawRois(ctx,lane,mask,sel,W,H,r,color){
  if(!lane)return;
  ctx.strokeStyle=color;ctx.lineWidth=1;
  for(const i of mask){
    const p=lane.centres[i];if(!sel.has(i)||!validPt(p,W,H))continue;
    ctx.strokeRect(Math.max(0,p[0]-r),Math.max(0,p[1]-r),Math.min(2*r+1,W-p[0]+r),Math.min(2*r+1,H-p[1]+r));
  }
}
function renderPanel(canvas,which,imgs,f){
  const ctx=canvas.getContext('2d'), W=f.width,H=f.height;
  if(canvas.width!==W||canvas.height!==H){canvas.width=W;canvas.height=H;}
  const bg=$(which+'Bg').value;
  ctx.fillStyle='#0c1217';ctx.fillRect(0,0,W,H);
  if(bg!=='none'&&imgs[bg])ctx.drawImage(imgs[bg],0,0,W,H);
  const sel=displayed(),mask=formSettings().of_joints;
  const isLeft=which==='left';
  if($('showNet').checked&&f.detected)skeleton(ctx,f.raw,'#ff6167',sel,W,H,2);
  if(!isLeft){
    if($('showIntegrated').checked){
      if(f.ready_a)skeleton(ctx,f.a.integrated,'#5edf9d',sel,W,H,2);
    }
    if($('showRois').checked&&f.ready_a)drawRois(ctx,f.a,mask,sel,W,H,f.roi,'#b579d2');
    if($('showArrows').checked&&f.ready_a)drawVectors(ctx,f.a,sel,mask,W,H,'#ffd25d');
  }
  if($('showFused').checked&&f.ready_a)skeleton(ctx,f.a.fused,'#47d9ed',sel,W,H,2);
  if($('showCompare').checked&&f.compare&&f.ready_b){
    skeleton(ctx,f.b.fused,'#ff95d6',sel,W,H,2);
    if(!isLeft&&$('showArrows').checked)drawVectors(ctx,f.b,sel,mask,W,H,'#ff9bdc');
  }
  if($('showTime').checked){
    ctx.fillStyle='#0e1723cb';ctx.fillRect(0,0,Math.min(W,308),27);
    ctx.fillStyle='#fff';ctx.font='12px monospace';
    ctx.fillText(`t=${f.time.toFixed(3)} s   detection age=${f.detected?f.detection_age.toFixed(3):'--'} s`,7,17);
  }
}
function renderRecordCanvas(){
  const lay=$('layout').value, solo=$('solo').value;
  const left=panels.left,right=panels.right;
  const w=left.width,h=left.height;
  const targetW=lay==='side'?w*2:w;
  const targetH=lay==='stack'?h*2:h;
  if(mainCanvas.width!==targetW)mainCanvas.width=targetW;
  if(mainCanvas.height!==targetH)mainCanvas.height=targetH;
  const ctx=mainCanvas.getContext('2d');
  if(lay==='side'){ctx.drawImage(left,0,0);ctx.drawImage(right,w,0);}
  else if(lay==='stack'){ctx.drawImage(left,0,0);ctx.drawImage(right,0,h);}
  else ctx.drawImage(solo==='left'?left:right,0,0);
}
function render(){
  configureLayout();
  if(currentFrame&&images){
    $('emptyStage').hidden=true;
    $('videoStage').hidden=false;
    $('videoStage').style.setProperty('--source-ratio', `${currentFrame.width} / ${currentFrame.height}`);
    renderPanel(panels.left,'left',images,currentFrame);
    renderPanel(panels.right,'right',images,currentFrame);
    renderRecordCanvas();
    updateNumerics(currentFrame);
  }
}
async function decodeJpeg(base64){
  return new Promise((resolve,reject)=>{
    const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('JPEG decode failed'));
    im.src='data:image/jpeg;base64,'+base64;
  });
}
let lastAcceptedFrame=-1;
async function acceptFrame(f){
  const timestamp=f.time;
  if(lastAcceptedFrame > timestamp+0.001 && lastAcceptedFrame-timestamp>1){chartHistory=[];}
  lastAcceptedFrame=timestamp;
  const decoded=await Promise.all(['eros','events','sae'].map(x=>decodeJpeg(f[x])));
  currentFrame=f;images={eros:decoded[0],events:decoded[1],sae:decoded[2]};
  const j=+$('selectedJoint').value;const va=f.a.velocity[j];const vb=f.b.velocity[j];
  chartHistory.push({t:f.time,x:va[0],y:va[1],mag:Math.hypot(va[0],va[1]),magB:Math.hypot(vb[0],vb[1])});
  if(chartHistory.length>450)chartHistory.shift();
  render();drawChart();
}
async function frameLoop(){
  while(true){
    try{
      const r=await request('/api/frame?after='+frameSeq);
      if(r.state){updateState(r.state);}
      if(typeof r.seq==='number')frameSeq=r.seq;
      if(r.frame){await acceptFrame(r.frame);}
      else await new Promise(res=>setTimeout(res,90));
    }catch(e){await new Promise(res=>setTimeout(res,1200));}
  }
}
async function statusLoop(){
  while(true){try{updateState(await request('/api/status'));}catch(e){}
    await new Promise(r=>setTimeout(r,500));}
}
function updateNumerics(f){
  const j=+$('selectedJoint').value;
  $('jointHeader').textContent=jointNames[j];
  if(!f.ready_a){['vx','vy','vmag','vmagB'].forEach(id=>$(id).textContent='—');return;}
  const v=f.a.velocity[j], vb=f.b.velocity[j];
  $('vx').textContent=v[0].toFixed(1);$('vy').textContent=v[1].toFixed(1);
  $('vmag').textContent=Math.hypot(v[0],v[1]).toFixed(1);
  $('vmagB').textContent=f.compare?Math.hypot(vb[0],vb[1]).toFixed(1):'—';
}
function drawChart(){
  const canvas=$('velocityChart'),ctx=canvas.getContext('2d');const W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle='#121920';ctx.fillRect(0,0,W,H);
  if(chartHistory.length<2)return;
  const newest=chartHistory[chartHistory.length-1].t;
  const rows=chartHistory.filter(a=>a.t>=newest-4);
  if(rows.length<2)return;
  let highest=20;
  for(const a of rows)highest=Math.max(highest,Math.abs(a.x),Math.abs(a.y),a.mag,a.magB);
  highest=Math.ceil(highest/20)*20;
  for(let i=0;i<=4;i++){
    const y=12+(H-30)*i/4;
    ctx.strokeStyle='#31424a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(W-8,y);ctx.stroke();
    ctx.fillStyle='#8498a5';ctx.font='10px monospace';
    ctx.fillText((highest-(2*highest)*i/4).toFixed(0),3,y+4);
  }
  const t0=Math.max(0,newest-4),scaleX=t=>35+(W-43)*(t-t0)/Math.max(.01,newest-t0);
  const scaleY=v=>12+(H-30)*(highest-v)/(2*highest);
  const lines=[['x','#75cffa'],['y','#f0b0a0'],['mag','#e7cd71']];
  if(currentFrame?.compare)lines.push(['magB','#f0a8dd']);
  for(const [key,color] of lines){
    ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.beginPath();
    rows.forEach((p,i)=>{let x=scaleX(p.t),y=scaleY(p[key]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
  }
}
$('screenshot').onclick=()=>{
  if(!currentFrame){notify('Open a sample first',true);return;}
  renderRecordCanvas();mainCanvas.toBlob(blob=>{
    if(!blob){notify('Screenshot failed',true);return;} download(blob,`of_sanity_t${currentTime.toFixed(2)}.png`);
  },'image/png');
};
function download(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),6000);}
$('record').onclick=async()=>{
  if(recording){if(mediaRecorder&&mediaRecorder.state==='recording')mediaRecorder.stop();return;}
  if(!currentFrame){notify('Open a recording first',true);return;}
  renderRecordCanvas();
  if(typeof MediaRecorder==='undefined'){notify('MediaRecorder is unsupported in this browser',true);return;}
  const kinds=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];
  const format=kinds.find(k=>MediaRecorder.isTypeSupported(k));
  if(!format){notify('Browser does not support WebM capture',true);return;}
  captured=[];const stream=mainCanvas.captureStream(+$('fps').value);
  mediaRecorder=new MediaRecorder(stream,{mimeType:format,videoBitsPerSecond:3500000});
  mediaRecorder.ondataavailable=e=>{if(e.data.size)captured.push(e.data);};
  mediaRecorder.onstop=async()=>{
    recording=false;$('record').textContent='Record MP4';
    const blob=new Blob(captured,{type:'video/webm'});captured=[];
    if(!blob.size){notify('No frames were recorded',true);return;}
    notify('Converting browser recording to MP4…');
    try{
      const response=await fetch('/api/convert',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:blob});
      if(!response.ok){const result=await response.json();throw new Error(result.error||'Conversion failed');}
      download(await response.blob(),`of_sanity_recording_${Date.now()}.mp4`);
      notify('MP4 ready');
    }catch(e){notify(e.message,true);download(blob,`of_sanity_fallback_${Date.now()}.webm`);}
    stream.getTracks().forEach(t=>t.stop());
  };
  mediaRecorder.start(1000);recording=true;$('record').textContent='Stop recording';
  notify('Recording the current browser playback. Click again to finish.');
};

makeJoints();configureLayout();$('emptyStage').hidden=false;
request('/api/status').then(s=>{if(s.settings){fillForm(s.settings);settingsLoaded=true;activeGeneration=s.generation;frameSeq=-1;}updateState(s);}).catch(()=>{});
frameLoop();statusLoop();
