const {P,ease,fmt,I}=require('./draw.cjs');
function sampling(d,s,T){
 const k=Math.min(3,Math.floor(s/3.5)),step=T.sampler_steps[k];
 const p=s>=14?1:ease((s-k*3.5-.7)/1.6),previous=T.sampler_states[k];
 const state=previous.map((v,i)=>v+p*step.delta[i]);
 d.text('Inference: apply each predicted velocity to the state',440,211,26);
 d.text('x(r) = x(t) − (t − r) U(x(t), y, t, r, h)',440,267,29);
 d.text('Step '+(k+1)+' / 4     t = '+step.t.toFixed(2)+' → r = '+step.r.toFixed(2),440,316,23,P.time);
 const times=[1,.75,.5,.25,0];d.line([[470,369],[1310,369]],P.line,3);
 for(let j=0;j<5;j++){const x=470+j*210;d.dot(x,369,j<=k?P.time:P.line,5);d.text(times[j].toFixed(2),x,344,17,P.muted,'center');}
 d.dot(470+(k+p)*210,369,P.track,8);d.grid(state);
 d.text('Start '+fmt(previous[I])+'     Predicted U '+fmt(step.velocity[I]),440,801,27);
 d.text('Δx = −0.25 × '+fmt(step.velocity[I])+' = '+fmt(step.delta[I]),440,850,28);
 d.text('New state: '+fmt(previous[I])+' + ('+fmt(step.delta[I])+') = '+fmt(step.next[I]),440,899,27,P.track);
 d.note('Computed toy updates · no clean target enters the sampler');
}
module.exports={sampling};
