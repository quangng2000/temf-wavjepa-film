const fs=require('node:fs'),path=require('node:path'),{spawn}=require('node:child_process');
const {once}=require('node:events'),{createCanvas}=require('@napi-rs/canvas');
const {P,tools,sidebar}=require('./draw.cjs'),T=require('./trace.json');
const F=require('./scenes-forward.cjs'),O=require('./scenes-objective.cjs');
const {sampling}=require('./scenes-sampling.cjs');
const phases=[[10,'Build a training state',F.pair],[10,'Extract CNN features',F.cnn],
 [14,'Condition on flow time',F.time],[12,'Apply FiLM from frozen context',F.film],
 [10,'Predict interval velocity',O.velocity],[18,'Build the full JVP target',O.jvp],
 [12,'Update trainable weights',O.update],[18,'Sample without the clean target',sampling]];
const W=1440,H=1080,FPS=24,DURATION=phases.reduce((s,p)=>s+p[0],0);
const canvas=createCanvas(W,H),d=tools(canvas.getContext('2d'));
function paint(seconds){
 let i=0,s=seconds;while(i<phases.length-1&&s>=phases[i][0])s-=phases[i++][0];
 d.rect(0,0,W,H,P.bg);sidebar(d,T,i===7);phases[i][2](d,s,T);
 d.text('One example: FiLM + TEMF in motion',58,49,31);
 d.text((i+1)+' / 8 · '+phases[i][1],58,89,25,P.muted);
 d.line([[58,116],[1382,116]],P.line,2);d.line([[58,116],[58+1324*seconds/DURATION,116]],P.context,4);
 d.text('Numerical illustration · sCM-style conditioning + TEMF velocity objective',58,142,18,P.muted);
 d.line([[58,1013],[1382,1013]],P.line,1);
 d.rect(58,1041,9,9,P.plus);d.text('Positive',78,1046,18,P.muted);
 d.rect(175,1041,9,9,P.minus);d.text('Negative',195,1046,18,P.muted);
 d.outline(298,1040,11,11,P.track,2);d.text('Tracked cell',321,1046,18,P.muted);
 d.text('Toy model · not a trained speech model',1382,1046,18,P.muted,'right');
 return canvas;
}
function metadata(){
 let time=0;return {width:W,height:H,fps:FPS,duration_seconds:DURATION,frames:DURATION*FPS,
 chapters:phases.map(([duration,title])=>{const item={start:time,duration,title};time+=duration;return item;}),
 checks:T.checks,scope:'A tiny synthetic numerical example; context is a frozen stand-in, not wavJEPA output.'};
}
async function render(out){
 const file=path.join(out,'film-temf-scm-motion.mp4');
 const proc=spawn(process.env.FFMPEG_BINARY||'ffmpeg',['-hide_banner','-loglevel','error','-y',
  '-f','rawvideo','-pix_fmt','rgba','-s',`${W}x${H}`,'-r',String(FPS),'-i','pipe:0',
  '-an','-c:v','libx264','-threads','3','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',file],
  {stdio:['pipe','ignore','inherit']});
 const completion=new Promise((resolve,reject)=>{proc.on('error',reject);proc.on('close',code=>code===0?resolve():reject(new Error('ffmpeg exit '+code)));});
 let pipeError;proc.stdin.on('error',error=>{pipeError=error;});
 for(let n=0;n<DURATION*FPS;n++){
  if(pipeError)throw pipeError;const frame=paint(n/FPS).getContext('2d').getImageData(0,0,W,H).data;
  if(!proc.stdin.write(Buffer.from(frame.buffer,frame.byteOffset,frame.byteLength)))await once(proc.stdin,'drain');
  if(n%(FPS*10)===0)process.stdout.write(`Rendered ${n/FPS}s / ${DURATION}s\n`);
 }proc.stdin.end();await completion;fs.writeFileSync(path.join(out,'metadata.json'),JSON.stringify(metadata(),null,2)+'\n');
 console.log(file);
}
async function main(){
 const out=path.resolve(process.env.ANIMATION_OUTPUT||path.join(__dirname,'../../outputs/film-temf-scm-v2'));
 fs.mkdirSync(out,{recursive:true});
 if(process.argv.includes('--previews')){
  for(const t of [6,16,28,41,51,60,65,70,80,92,102])fs.writeFileSync(path.join(out,`preview-${t}.png`),paint(t).toBuffer('image/png'));
  console.log(out);return;
 }await render(out);
}
if(require.main===module)main().catch(error=>{console.error(error);process.exitCode=1;});
module.exports={paint,metadata};
