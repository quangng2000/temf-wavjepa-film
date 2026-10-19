const {P,clamp,ease,lerp,fmt,I}=require('./draw.cjs');
function pair(d,s,T){
 const t=1-.2*ease(s/4),state=T.clean.map((v,i)=>(1-t)*v+t*T.prior[i]);
 d.text('Construct the training state',440,211,29);
 d.text('x(t) = (1 − t)c + t z',440,271,31);d.text('z · Gaussian prior',1090,210,21,P.muted);
 d.grid(T.prior,1090,245,250,100,12,24,false);
 d.text('Current t = '+t.toFixed(2)+'     Destination r = 0.20',440,359,25,P.time);
 d.grid(state);d.flow([[328,797],[398,797],[398,590],[430,590]],s,P.plus);
 d.text('Tracked coefficient',440,807,22,P.track);
 d.text('(1 − '+t.toFixed(2)+') × '+fmt(T.clean[I])+' + '+t.toFixed(2)+' × '+fmt(T.prior[I]),440,858,27);
 d.text('x(t) = '+fmt(state[I]),440,906,31,P.track);
 d.note('Clean c builds the training state. Observed y stays separate.');
}
function cnn(d,s,T){
 const progress=clamp(s/6),column=Math.min(23,Math.floor(progress*24));
 const values=T.state.map((v,i)=>i%24<=column?T.before.features[i]:v);
 d.text('Convolutions read both x(t) and observed y',440,212,27);
 d.grid(T.weights_before['conv.weight'].slice(0,9),440,260,105,90,3,3,false,4);
 d.grid(T.weights_before['conv.weight'].slice(9,18),585,260,105,90,3,3,false,4);
 d.text('x kernel',440,367,19,P.muted);d.text('y kernel',585,367,19,P.muted);
 d.text('Shared kernel → feature channel 1',752,304,26,P.plus);
 d.grid(values);if(s<6)d.outline(440+Math.min(21,column)*37.5,400+4*350/12,112.5,350/4,P.plus,4);
 d.flow([[328,317],[398,317],[398,575],[430,575]],s,P.plus);
 d.text('Same location · new representation',440,809,23,P.track);
 d.text('x(t): '+fmt(T.state[I])+'   →   CNN feature F: '+fmt(T.before.features[I]),440,866,29);
 d.note('The gold cell follows this location as the representation changes.');
}
function time(d,s,T){
 const p=ease(s/5),q=ease((s-5)/5),F=T.before.features.slice(0,288),N=T.before.normalized.slice(0,288),G=T.before.timed.slice(0,288);
 const values=s<5?lerp(F,N,p):lerp(N,G,q);
 d.text('t = 0.80, r = 0.20 → fixed sin/cos → time MLP',440,210,26,P.time);
 d.grid(T.before.embedding,440,248,900,95,4,32,false);
 d.text('32 fixed angular frequencies per time · 1 down to 0.0001',440,365,20,P.muted);
 d.grid(values);d.flow([[1355,297],[1355,390]],s,P.time);
 d.text(s<5?'GroupNorm within each frame':'Normalized time controls scale and shift the features',440,796,24,P.time);
 d.text('Norm(F) = '+fmt(T.before.normalized[I]),440,841,26);
 d.text(fmt(T.before.normalized[I])+' × '+fmt(T.before.scale[0])+' + ('+fmt(T.before.bias[0])+') = '+fmt(T.before.timed[I]),440,888,28,P.track);
 d.note('G = GroupNorm(F) × pnorm(time scale) + pnorm(time bias)');
}
function film(d,s,T){
 const G=T.before.timed.slice(0,288),gamma=T.before.gamma[0],beta=T.before.beta[0];
 const scaled=G.map(v=>v*gamma),output=scaled.map(v=>v+beta);
 const values=s<5?lerp(G,scaled,ease(s/5)):lerp(scaled,output,ease((s-5)/4));
 d.text('Frozen h → trainable FiLM head → channel controls',440,211,26,P.context);
 for(let k=0;k<4;k++){const x=440+k*230;d.text('Channel '+(k+1),x,266,21,k===0?P.track:P.context);d.text('γ '+fmt(T.before.gamma[k],2),x,303,24,P.context);d.text('β '+fmt(T.before.beta[k],2),x,341,24,P.context);}
 d.grid(values);d.flow([[328,529],[394,529],[394,303],[430,303]],s,P.context);
 d.text(s<5?'First multiply by γ':'Then add β',440,801,26,P.context);
 d.text(fmt(G[I])+' × '+fmt(gamma)+' = '+fmt(scaled[I]),440,851,28);
 d.text(fmt(scaled[I])+' + '+fmt(beta)+' = '+fmt(output[I]),440,899,30,P.track);
 d.note('G′ = γ(h) × G + β(h) · both scale and offset affect the result');
}
module.exports={pair,cnn,time,film};
