const P={bg:'#0e1826',ink:'#edf2fa',muted:'#aebaca',line:'#405365',plus:'#58bfff',minus:'#f38e99',context:'#6bd8bc',time:'#b5a1ff',track:'#ffd17c'};
const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x);};
const lerp=(a,b,p)=>a.map((v,i)=>v+(b[i]-v)*p);
const fmt=(v,n=4)=>(v>=0?'+':'')+v.toFixed(n),I=130;
function tools(c){
 const text=(s,x,y,size=25,color=P.ink,align='left')=>{c.font=size+'px Arial';c.fillStyle=color;c.textAlign=align;c.textBaseline='middle';c.fillText(s,x,y);};
 const line=(pts,color=P.line,width=2)=>{c.beginPath();pts.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle=color;c.lineWidth=width;c.stroke();};
 const rect=(x,y,w,h,color)=>{c.fillStyle=color;c.fillRect(x,y,w,h);};
 const outline=(x,y,w,h,color=P.track,width=3)=>{c.strokeStyle=color;c.lineWidth=width;c.strokeRect(x,y,w,h);};
 const dot=(x,y,color=P.track,r=5)=>{c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fillStyle=color;c.fill();};
 const grid=(values,x=440,y=400,w=900,h=350,rows=12,cols=24,tracked=true,boost=1)=>{
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
   const v=values[row*cols+col]||0;c.globalAlpha=.10+.9*Math.tanh(Math.abs(v*boost)*1.6);
   rect(x+col*w/cols,y+row*h/rows,w/cols-2.5,h/rows-2.5,v>=0?P.plus:P.minus);
  }c.globalAlpha=1;
  if(tracked&&rows===12&&cols===24)outline(x+10*w/cols-1,y+5*h/rows-1,w/cols,h/rows,P.track,3);
 };
 const flow=(pts,s,color=P.context)=>{
  line(pts);const lengths=pts.slice(1).map((b,i)=>Math.hypot(b[0]-pts[i][0],b[1]-pts[i][1]));
  for(let k=0;k<5;k++){let pos=((s*.25+k/5)%1)*lengths.reduce((a,b)=>a+b,0);
   for(let j=0;j<lengths.length;j++){if(pos<=lengths[j]){const p=pos/lengths[j];dot(pts[j][0]+p*(pts[j+1][0]-pts[j][0]),pts[j][1]+p*(pts[j+1][1]-pts[j][1]),color);break;}pos-=lengths[j];}
  }
 };
 const note=(s,y=959,color=P.muted)=>text(s,890,y,22,color,'center');
 const wave=(v,x,y,w,h,color=P.plus,progress=1,peak=null)=>{
  peak=peak||Math.max(1e-8,...v.map(Math.abs));line([[x,y+h/2],[x+w,y+h/2]],P.line,1);
  const count=Math.max(2,Math.floor(w)),pts=[];
  for(let j=0;j<=count*clamp(progress);j++){const index=Math.min(v.length-1,Math.floor(j/count*(v.length-1)));pts.push([x+j/count*w,y+h/2-v[index]/peak*h*.46]);}
  line(pts,color,2);
 };
 return {c,text,line,rect,outline,dot,grid,flow,note,wave};
}
function sidebar(d,T,inference){
 d.text('PAIR A · '+(inference?'INFERENCE':'TRAINING'),58,173,21,P.context);
 d.text('Observed y · real plane',58,213,22);d.grid(T.observed,58,244,270,145);
 d.text('Frozen context h',58,447,23,P.context);d.grid(T.context,58,480,270,95,4,24,false);
 d.text('JEPA features illustrated',58,606,18,P.muted);
 d.outline(303,422,17,18,P.context,2);d.line([[306,422],[306,416],[317,416],[317,422]],P.context,2);
 d.text(inference?'Clean target unavailable':'Training-only clean c',58,692,22,inference?P.muted:P.ink);
 if(!inference)d.grid(T.clean,58,724,270,145);
 else{d.outline(58,724,270,145,P.line,1);d.text('Not passed to sampler',193,797,21,P.muted,'center');}
 d.text('Follow the gold cell',58,917,22,P.track);d.text('Bin 6 · frame 11',58,948,20,P.muted);
 d.text('First 12 / 257 bins shown',58,979,17,P.muted);
 d.line([[366,173],[366,995]],P.line,1);
}
module.exports={P,clamp,ease,lerp,fmt,I,tools,sidebar};
