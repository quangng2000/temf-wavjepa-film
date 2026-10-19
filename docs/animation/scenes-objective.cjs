const {P,clamp,ease,lerp,fmt,I}=require('./draw.cjs');
function velocity(d,s,T){
 const output=T.before.velocity,q=.9-.8/(1+Math.exp(-8*(.5-.6)));
 d.text('Predict average velocity over the selected interval',440,211,26);
 d.text('TEMF schedule at 50% progress: '+(q*100).toFixed(1)+'% local / '+((1-q)*100).toFixed(1)+'% interval',440,264,22,P.muted);
 d.line([[480,326],[1290,326]],P.line,4);d.dot(480+.2*810,326,P.time,9);d.dot(480+.8*810,326,P.time,9);
 d.text('r = 0.20',642,364,23,P.time,'center');d.text('t = 0.80',1128,364,23,P.time,'center');
 d.grid(lerp(T.before.modulated.slice(0,288),output,ease(s/4)));
 d.text('Velocity U = '+fmt(output[I]),440,807,29,P.track);
 d.text('Reverse step: '+fmt(T.state[I])+' − 0.60 × '+fmt(output[I]),440,860,27);
 d.text('would give '+fmt(T.state[I]-.6*output[I]),440,904,28);
 d.note('A positive velocity subtracts from a coefficient on a reverse step.');
}
function jvp(d,s,T){
 const U=T.before.velocity;
 if(s<7){
  d.text('Nudge the state and current time together',440,211,28);
  d.text('x → x + ε v_mix      t → t + ε      r stays fixed',440,264,25,P.time);
  d.text('ε = 0.0001 · same y and h',440,313,22,P.muted);
  d.text('Original prediction',440,366,24);d.text('Prediction after the nudge',920,366,24);
  d.grid(U,440,400,420,350);d.grid(T.nudged,920,400,420,350);
  d.text('U = '+fmt(U[I],7),440,799,27,P.track);d.text('U+ = '+fmt(T.nudged[I],7),920,799,27,P.track);
  d.text('(U+ − U) / ε ≈ '+fmt((T.nudged[I]-U[I])/T.epsilon,5),440,864,30);
  d.note('The small visible difference measures a directional derivative.');
 }else if(s<11){
  d.text('Full JVP includes both state and time change',440,211,27);
  d.text('J = D_x U · v_mix + ∂U/∂t',440,282,31,P.time);
  d.text('The destination time r is held fixed.',440,341,23,P.muted);d.grid(T.jvp);
  d.text('State contribution: '+fmt(T.jx[I],6),440,806,27);
  d.text('Time contribution: '+fmt(T.jt[I],6),440,853,27,P.time);
  d.text('J = '+fmt(T.jvp[I],6),440,902,31,P.track);
  d.note('Automatic JVP checked against a finite-difference calculation.');
 }else{
  d.text('Use the derivative to build the interval target',440,211,27);
  d.text('target = stopgrad[v_mix − (t − r) J]',440,281,31);
  d.text('Target held fixed during parameter backpropagation',440,341,22,P.track);
  d.grid(lerp(T.mixed,T.target,ease((s-11)/4)));
  d.text(fmt(T.mixed[I])+' − 0.60 × '+fmt(T.jvp[I])+' = '+fmt(T.target[I]),440,825,28,P.track);
  d.text('v_mix blends path velocity and predicted local velocity.',440,887,24,P.muted);
  d.note('J adjusts the velocity target for this interval; it is not an extra audio filter.');
 }
}
function update(d,s,T){
 const p=ease((s-4)/5),errors=lerp(T.error_before,T.error_after,p);
 d.text('Weighted velocity error → one SGD update',440,211,28);
 d.text('L = mean[w(t − r) × (U − target)²]',440,269,30);
 d.text('Target detached · θ ← θ − 0.003 × gradient',440,321,23,P.muted);
 d.grid(errors,440,400,900,350,12,24,true,20);
 d.text('Error map brightness ×20 · squared errors are nonnegative',440,368,20,P.muted);
 const groups=[['CNN','conv.weight'],['FiLM adapter','film.weight'],['Time MLP','time_mlp.0.weight']];
 for(let k=0;k<3;k++){
  const x=440+k*305,[name,key]=groups[k],gradient=T.gradients[key];
  d.text(name+(s<5?' gradient ×100':' · one weight'),x,803,20,P.track);
  if(s<5)d.grid(gradient.slice(0,24),x,833,270,60,2,12,false,100);
  else{
   const j=gradient.reduce((best,v,i)=>Math.abs(v)>Math.abs(gradient[best])?i:best,0);
   const old=T.weights_before[key][j],next=T.weights_after[key][j];
   d.text('Before '+fmt(old,8),x,842,21,P.muted);
   d.text('After  '+fmt(old+(next-old)*p,8),x,880,21,P.track);
  }
 }
 d.text(T.checks.fixed_target_loss_before.toFixed(8)+' → '+T.checks.fixed_target_loss_after.toFixed(8),440,934,30,P.track);
 d.note('Loss against this fixed target · one update does not establish convergence',984);
}
module.exports={velocity,jvp,update};
