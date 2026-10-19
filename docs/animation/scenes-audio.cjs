const {P,clamp,ease,lerp,fmt,I}=require('./draw.cjs');
const peak=v=>Math.max(1e-8,...v.map(Math.abs));
const reveal=(v,p)=>v.map((x,i)=>i%24<Math.ceil(p*24)?x:0);
function opening(d,s,T){
 const p=clamp(s/10),col=Math.min(23,Math.floor(p*24));
 d.text('One damaged waveform → two complementary representations',58,205,29);
 d.text('Synthetic audio · 16 kHz · same recording in both branches',58,251,23,P.muted);
 d.wave(T.audio.damaged_wave,78,296,1270,150,P.plus);
 const cursor=78+col*T.audio.hop/T.audio.damaged_wave.length*1270,half=T.audio.fft/T.audio.damaged_wave.length*635;
 const left=Math.max(78,cursor-half),right=Math.min(1348,cursor+half);d.outline(left,290,right-left,162,P.track,2);
 d.flow([[714,462],[714,486],[350,486],[350,533]],s,P.plus);
 d.flow([[714,462],[714,486],[1060,486],[1060,533]],s,P.context);
 d.text('STFT → magnitude compression → y',78,566,26,P.plus);
 d.text('Frozen wavJEPA encoder → context h',785,566,26,P.context);
 d.grid(reveal(T.observed.slice(0,288),p),78,605,580,210);
 d.grid(reveal(T.context,p),785,605,555,210,4,24,false);
 d.text('Real plane shown · imaginary plane is also retained',78,846,20,P.muted);
 d.text('Context values illustrated by a fixed projection',785,846,20,P.muted);
 d.text('Signal coefficients retain magnitude and phase.',78,900,22);
 d.text('Context conditions the model through FiLM.',785,900,22);
 d.text('STFT is computed here. The real wavJEPA encoder is not run in this toy.',720,974,23,P.muted,'center');
 d.outline(1345,549,17,18,P.context,2);d.line([[1348,549],[1348,541],[1359,541],[1359,549]],P.context,2);
}
function target(d,s,T){
 const p=clamp(s/6),col=Math.min(23,Math.floor(p*24));
 d.text('Training adds an aligned clean waveform',440,211,28);
 d.wave(T.audio.clean_wave,440,253,900,100,P.plus);
 const center=440+col*T.audio.hop/T.audio.clean_wave.length*900,half=T.audio.fft/T.audio.clean_wave.length*450;
 const left=Math.max(440,center-half),right=Math.min(1340,center+half);d.outline(left,248,right-left,111,P.track,2);
 d.text('Same STFT + compression → clean coefficients c',440,374,24);
 d.grid(reveal(T.clean,p));
 d.text('The waveform pair covers the same audio samples.',440,804,24);
 d.text('c supplies the training path and target information.',440,855,24);
 d.text('At inference, only the damaged recording is available.',440,907,24,P.context);
 d.note('Transform round-trip checked numerically before animating.');
}
function decompression(d,s,T){
 const coefficients=T.sampler_states.at(-1),p=ease(s/4);
 d.text('Final state → inverse magnitude compression',440,211,27);
 d.text('Restore magnitude with power 1 / 0.3; preserve phase',440,267,24,P.muted);
 d.text('Real plane',440,361,24);d.text('Imaginary plane',920,361,24);
 d.grid(lerp(coefficients.slice(0,288),T.audio.spectrum_real,p),440,400,420,350);
 d.grid(lerp(coefficients.slice(288,576),T.audio.spectrum_imag,p),920,400,420,350);
 d.text('Predicted compressed value: '+fmt(coefficients[I]),440,817,26,P.track);
 d.text('Uncompressed real component: '+fmt(T.audio.spectrum_real[I]),440,869,26);
 d.note('Both complex planes are used by the inverse STFT.');
}
function overlap(d,s,T){
 d.text('Inverse FFT → window → overlap and add',440,211,28);
 d.text('Three neighboring frames shown; all 24 reconstruct the clip',440,265,23,P.muted);
 const parts=T.audio.contributions,amplitude=peak(parts),colors=[P.plus,P.context,P.time];
 for(let k=0;k<3;k++){
  const x=440+k*225,y=332+k*110,part=parts.slice(k*512,(k+1)*512);
  d.text('Frame '+(T.audio.frame_ids[k]+1),x,y-15,20,colors[k]);
  d.wave(part,x,y,450,85,colors[k],clamp((s-5-k*.5)/2),amplitude);
  d.flow([[x+225,y+88],[x+225,698],[890,698],[890,722]],s,colors[k]);
 }
 d.wave(T.audio.output_wave,440,744,900,145,P.track,clamp((s-7)/4));
 d.text('Add overlapping windows, then divide by their summed squared weights.',440,929,21,P.muted);
 d.note('The explicit overlap-add result matches the inverse-STFT output.',979);
}
function output(d,s,T){
 d.text('Waveform out: the result of the computed toy steps',440,211,27);
 const scale=Math.max(peak(T.audio.damaged_wave),peak(T.audio.output_wave));
 d.text('Damaged input',440,298,25,P.plus);d.wave(T.audio.damaged_wave,440,337,900,175,P.plus,1,scale);
 d.text('Reconstructed toy output',440,598,25,P.track);d.wave(T.audio.output_wave,440,640,900,175,P.track,1,scale);
 const p=((s-12)/5)%1;d.line([[440+p*900,330],[440+p*900,825]],P.context,2);
 d.text('Same vertical scale · actual output of the sampler + inverse STFT',440,877,22,P.muted);
 d.text('One toy weight update does not establish restoration quality.',440,931,23,P.muted);
}
function reconstruction(d,s,T){
 if(s<5)decompression(d,s,T);else if(s<12)overlap(d,s,T);else output(d,s,T);
}
module.exports={opening,target,reconstruction};
