import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root=path.resolve('..'),out=path.join(root,'output');await fs.mkdir(out,{recursive:true});
const srcs=[process.env.DATASET_TEMPLATE_PPTX,process.env.METHOD_TEMPLATE_PPTX];
if(srcs.some(x=>!x)) throw new Error('Set DATASET_TEMPLATE_PPTX and METHOD_TEMPLATE_PPTX to licensed reference decks');
const trace=JSON.parse(await fs.readFile('trace.json','utf8'));
const C={blue:'#4D8AB9',ink:'#243D51',orange:'#D98246',green:'#6EA64A',teal:'#328C89',violet:'#9564C6',gray:'#647584'};
function shape(s,x,y,w,h,fill='none',stroke='none',geom='rect',width=1){return s.shapes.add({geometry:geom,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width,style:'solid'}})}
function text(s,txt,x,y,w,h,size=18,color=C.ink,bold=false,align='left',font='Arial'){let z=shape(s,x,y,w,h);z.text=txt;z.text.style={fontSize:size,typeface:font,color,bold,alignment:align,verticalAlignment:'middle',wrap:'none',insets:0};return z;}
function line(s,x1,y1,x2,y2,color=C.gray,width=1.5){return s.shapes.add({geometry:'custom',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.max(1,Math.abs(x2-x1)),height:Math.max(1,Math.abs(y2-y1))},fill:'none',line:{fill:color,width,style:'solid'},customPaths:[{width:Math.max(1,Math.abs(x2-x1)),height:Math.max(1,Math.abs(y2-y1)),commands:[{moveTo:{x:x1-Math.min(x1,x2),y:y1-Math.min(y1,y2)}},{lineTo:{x:x2-Math.min(x1,x2),y:y2-Math.min(y1,y2)}}]}]});}
function arrow(s,x,y,w,h,color){return shape(s,x,y,w,h,color,'none','rightArrow');}
async function img(s,n,x,y,w,h){let b=await fs.readFile(`episode/frame_${n}.png`);return s.images.add({blob:b,contentType:'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt:`LIBERO source demonstration, approximate control step ${n}`});}
function token(s,x,y,w,h,color,label=''){shape(s,x,y,w,h,color,color,'roundRect',1);if(label)text(s,label,x,y,w,h,14,'#FFFFFF',true,'center');}
function curve(s,x,y,w,h,values,color){let max=Math.max(...values),min=Math.min(...values);let pts=values.map((v,i)=>({x:i*w/(values.length-1),y:h-(v-min)/(max-min||1)*h}));s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width:2,style:'solid'},customPaths:[{width:w,height:h,commands:[{moveTo:pts[0]},...pts.slice(1).map(p=>({lineTo:p}))]}]});}
function actionplot(s,x,y,w,h){line(s,x,y+h,x+w,y+h,'#9CA5AC',1);line(s,x,y,x,y+h,'#9CA5AC',1);for(let j=0;j<3;j++)curve(s,x+6,y+8+j*9,w-12,h-22,trace.actions.map(a=>a[j]),[C.blue,C.orange,C.teal][j]);}
function edit(p,id,txt,pos,size){let z=p.resolve(id);if(txt!==null)z.text=txt;if(pos)z.position={left:pos[0],top:pos[1],width:pos[2],height:pos[3]};if(size)z.text.fontSize=size;return z;}
function axes(s,x,y,scale=1){line(s,x,y,x+38*scale,y,'#C95C59',2.5);line(s,x,y,x-20*scale,y+20*scale,'#409864',2.5);line(s,x,y,x,y-40*scale,'#5084C5',2.5);text(s,'x',x+40*scale,y-8,20,20,14,'#C95C59');text(s,'y',x-35*scale,y+15*scale,20,20,14,'#409864');text(s,'z',x-6,y-58*scale,20,20,14,'#5084C5');shape(s,x-4,y-4,8,8,C.teal,'none','ellipse');}
async function save(p,stem){const s=p.slides.items[0];await fs.writeFile(path.join(out,stem+'.png'),new Uint8Array(await(await p.export({slide:s,format:'png',scale:2})).arrayBuffer()));await fs.writeFile(stem+'.layout.json',await(await s.export({format:'layout'})).text());await(await PresentationFile.exportPptx(p)).save(path.join(root,'build',stem+'.draft.pptx'));try{await fs.writeFile(path.join(out,stem+'.pdf'),new Uint8Array(await(await s.export({format:'pdf'})).arrayBuffer()));}catch(e){console.log('PDF',e.message)} }
// Dataset: retain both template rows and their stage frames.
{
const p=await PresentationFile.importPptx(await FileBlob.load(srcs[0]));p.slides.keep([1]);const s=p.slides.items[0];
const layout=JSON.parse(await fs.readFile('ref1_slide2_layout.json','utf8'));
const keep=new Set(['sh/ved0jihs','sh/9072xkry','sh/9wnqhczy','sh/mtwrmxg7','sh/nu58f2hs','sh/wjy9sry9','sh/xk7qlczu','sh/ahwrqhgj','sh/bip8jmho','sh/cnupgny5','sh/do3q9szq','sh/t8byxkn2','sh/s72xofmh','sh/76twva5c','sh/65kfmpor','sh/hcvy14ne','sh/gbmxszmt','im/0ruxkv2l','im/ipgze94z','sh/107uh0v6','sh/gzydovul','sh/v2lc36dw','sh/a1cba1wb','sh/0vq1o36x','sh/1wzix87i','sh/et8zmto7','sh/zuh0vypc','sh/c76hk361','sh/xgfil8zy']);
p.delete(layout.elements.filter(e=>!keep.has(e.id)).map(e=>e.id));
edit(p,'sh/ved0jihs','Episode preparation & annotation',[332,0,1000,36],30);
edit(p,'sh/9072xkry','Endpoint geometry + event keyframes replace the function body',[332,397,1000,34],25);
for(const [id,t] of [['sh/do3q9szq','Standardization'],['sh/76twva5c','Atomic Segmentation'],['sh/gbmxszmt','Clip to Typed Call'],['sh/v2lc36dw','Terminal Pose'],['sh/1wzix87i','Event Keyframes'],['sh/c76hk361','Action Alignment']])edit(p,id,t,null,25);
// source film mosaic
text(s,'Robot demonstrations',14,7,265,32,25,C.blue,true,'center');
for(const [n,x,y] of [[0,15,49],[91,149,49],[207,15,168],[294,149,168]])await img(s,n,x,y,125,104);
text(s,'“Put the bowl on the plate”',15,282,263,28,17,C.ink,true,'center');
text(s,'RGB  +  robot state  +  actions',15,315,263,26,17,C.gray,false,'center');
text(s,'Synchronized episode',15,346,263,26,18,C.blue,true,'center');
// standardization, preserve source robot silhouettes / axes image
text(s,'Embodiment & coordinate mapping',345,174,247,29,16,C.ink,false,'center');
text(s,'Base frame',348,216,75,26,16,C.blue,true,'center');text(s,'Gripper',437,216,76,26,16,C.blue,true,'center');text(s,'Joint order',519,216,76,26,16,C.blue,true,'center');
for(const x of [386,473,557]){line(s,x,245,x,270,C.blue);}
text(s,'Pose units     Open / close     Joint mapping',347,364,248,18,12,C.gray,false,'center');
// segmentation temporal strip
for(let i=0;i<3;i++){await img(s,[45,140,207][i],681+i*78,119,75,65);text(s,['start t','contact','finish c'][i],677+i*80,188,80,22,14,C.gray,false,'center');}
line(s,687,232,914,232,C.gray,2);for(let i=0;i<5;i++)line(s,687+i*56.5,226,687+i*56.5,238,C.gray,1);
for(const [x,w,t,c] of [[686,72,'approach','#D9E7F4'],[760,68,'close','#E8E6CC'],[830,84,'lift','#DBEDD9']]){shape(s,x,247,w,31,c,'none');text(s,t,x,247,w,31,16,C.ink,false,'center');}
line(s,687,297,914,297,C.blue,2);line(s,687,287,687,297,C.blue,2);line(s,914,287,914,297,C.blue,2);
text(s,'One pick primitive',685,304,233,27,20,C.blue,true,'center');
text(s,'Completion event defines c',680,343,240,24,17,C.gray,false,'center');
// Explicit semantic conversion, including evidence and validation.
text(s,'1  Infer operation from state changes',1033,119,310,25,18,C.blue,true);
text(s,'approach + close + lifted object = pick',1033,147,310,23,16,C.ink);
text(s,'2  Bind references and contract',1033,177,310,25,18,C.blue,true);
text(s,'scene: bowl    arm: right    result: held',1033,205,310,23,16,C.ink);
text(s,'3  Serialize validated string slots',1033,235,310,25,18,C.blue,true);
text(s,'pick(object="bowl",\n target="above_table",\n arm="right", completion="held")',1033,266,311,70,15,C.green,true,'left','Courier New');
text(s,'Ambiguous evidence goes to review',1033,350,310,24,17,C.gray);
// bottom input sample
text(s,'Current observation',14,440,264,30,24,C.orange,true,'center');await img(s,45,17,489,257,209);
text(s,'t = start of this training window',14,706,264,26,17,C.gray,false,'center');
text(s,'Split episodes before sampling',14,749,265,25,17,C.orange,true,'center');
// pose target
await img(s,207,349,520,134,108);axes(s,542,595,0.8);
text(s,'End state: object held',345,637,248,24,18,C.ink,true,'center');
text(s,'Query: (pᶠ, Rᶠ)',348,669,246,32,26,C.teal,true,'center');
text(s,'Where the primitive finishes',345,712,248,24,18,C.ink,false,'center');
text(s,'One pose per arm',342,746,252,22,15,C.gray,false,'center');
// Event-dependent visual logic, with different numbers of frames.
text(s,'pick: pregrasp / close / lift',680,515,247,23,17,C.blue,true,'center');
for(let i=0;i<3;i++){let x=680+i*81;await img(s,[140,158,207][i],x,543,77,70);text(s,['pregrasp','closed','lifted'][i],x,614,77,21,14,C.blue,false,'center');}
line(s,683,646,921,646,'#DCE3E7',1);
text(s,'move: endpoint',681,658,133,25,18,C.blue,true);
text(s,'Different atoms,\ndifferent events',681,693,134,45,16,C.ink);
await img(s,253,819,657,101,81);text(s,'arrived',820,739,100,20,14,C.blue,false,'center');
text(s,'Variable event count',681,763,243,17,14,C.gray,false,'center');
// action targets plus align record
 actionplot(s,1040,525,293,80);text(s,'Actions realize event progression',1040,612,295,24,18,C.violet,true,'center');
for(let i=0;i<9;i++)token(s,1043+i*32,650,26,25,i<6?'#B89CD6':'#DCE2E7',i<6?'a':'');
text(s,'valid actions',1043,680,180,24,16,C.violet);text(s,'mask boundary',1209,680,130,24,15,C.gray,false,'right');
line(s,1038,714,1338,714,'#DCE3E7',1);text(s,'{oₜ, sₜ, Pᵗˣᵗ, Xᶠ, Iᶠ, Aₜ}',1034,726,307,31,22,C.ink,true,'center');text(s,'Event roles, timestamps and validity masks',1034,760,307,18,14,C.gray,false,'center');
s.speakerNotes.text='Design reference: user-provided robot-data-processing-editable.pptx, editable slide 2. Original robot silhouettes and calibration illustration retained as visual reference assets. LIBERO frames are from local source demonstration put_the_bowl_on_the_plate__seed11/source.mp4, not AtomVLA predictions. The pick interval shown is an illustrative re-annotation of approach/closure/lift into one typed primitive. The action curves come from its source_trace.npz. Method content follows sec/3_method.tex in the current manuscript. No dataset size or new empirical result is claimed.';
await save(p,'AtomVLA_dataset_pipeline');}
// Method: preserve the original expert-band composition, shared-attention bar and program panel.
{
const p=await PresentationFile.importPptx(await FileBlob.load(srcs[1]));p.slides.keep([1]);const s=p.slides.items[0];const layout=JSON.parse(await fs.readFile('ref2_slide2_layout.json','utf8'));
const keep=new Set(['sh/9072xkry','sh/ozy1ofad','sh/b29kza94','sh/a10jqpsj','sh/w3i1sfa9','sh/on218vil','sh/nmt0fq10','sh/mlkj6l0f','sh/xcvil0j2','sh/wbm1cvix','sh/gful8jqt','sh/hg32ho7e','sh/uts3698n','sh/vulkfep8','sh/4julcjq5','sh/5k32lo7a']);
p.delete(layout.elements.filter(e=>!keep.has(e.id)).map(e=>e.id));
edit(p,'sh/9072xkry','AtomVLA Framework',[20,0,1328,40],31);
edit(p,'sh/ozy1ofad',null,[26,47,338,118]);edit(p,'sh/b29kza94',null,[26,47,338,28]);
edit(p,'sh/a10jqpsj','Discrete atomic call',[40,50,310,22],16);
edit(p,'sh/w3i1sfa9','pick(object="bowl",\n     target="above_table",\n     arm="right",\n     completion="held")',[40,80,314,78],15);
text(s,'Terminal pose',380,48,170,25,19,C.teal,true,'center');axes(s,416,130,0.65);text(s,'pᶠ, Rᶠ',458,84,80,33,23,C.teal,true,'center');text(s,'one per arm',446,126,98,23,15,C.gray,false,'center');
text(s,'Primitive-specific imagined keyframes',564,48,434,25,20,C.blue,true,'center');
for(let i=0;i<3;i++){let x=564+i*108;await img(s,[140,158,207][i],x,83,99,79);text(s,['pregrasp','closed','lifted'][i],x,163,99,20,14,C.blue,false,'center');}
line(s,888,85,888,181,'#CBD5DF',1);
await img(s,253,898,83,99,79);text(s,'move: end',898,163,99,20,14,C.blue,false,'center');
text(s,'Dense action chunk',1022,48,314,25,20,C.violet,true,'center');actionplot(s,1035,88,280,71);text(s,'motion + gripper commands',1025,165,310,21,16,C.gray,false,'center');
// expert boxes in the original horizontal band
edit(p,'sh/on218vil',null,[26,224,338,139]);edit(p,'sh/nmt0fq10','Program Expert',[26,233,338,34],27);edit(p,'sh/mlkj6l0f','Name + string arguments',[26,271,338,29],19);
shape(s,374,224,178,139,'#DCEFED','#6BA8A3','roundRect');text(s,'Action Query',380,235,166,31,24,C.ink,true,'center');text(s,'Terminal geometry',380,273,166,24,17,C.ink,false,'center');
edit(p,'sh/xcvil0j2',null,[562,224,440,139]);edit(p,'sh/wbm1cvix','Future Imagination',[562,233,440,34],27);edit(p,'sh/gful8jqt','Event keyframes replace the procedural body',[562,273,440,26],18);
edit(p,'sh/hg32ho7e',null,[1012,224,330,139]);edit(p,'sh/uts3698n','Action Expert',[1012,233,330,34],27);edit(p,'sh/vulkfep8','Continuous control',[1012,273,330,26],20);
// Attention band must sit above newly added query block.
p.delete(['sh/4julcjq5','sh/5k32lo7a']);shape(s,17,312,1333,49,'#E8E8E8','#8F8F8F','roundRect');text(s,'Shared MoT attention   [ context ; program ; Query / state ; image tokens ; action tokens ]',25,318,1317,35,24,'#30343A',false,'center');
// output tokens, label transfer from program to query
for(let i=0;i<5;i++)token(s,62+i*57,190,43,25,'#8EBB70');for(let i=0;i<3;i++)token(s,398+i*45,190,34,25,C.teal,'Q');
for(let i=0;i<6;i++)token(s,613+i*54,190,37,25,'#84ACD3');for(let i=0;i<5;i++)token(s,1051+i*54,190,37,25,'#B392D7');
arrow(s,350,191,38,17,C.green);
// input token rows
for(let i=0;i<4;i++){token(s,45+i*46,379,32,27,i<2?'#E9B85E':'#9DA6AD');}
text(s,'VLM context Hₜ',34,412,196,28,22,C.ink,true,'center');text(s,'current RGB + instruction',27,445,213,22,16,C.gray,false,'center');
for(let i=0;i<2;i++)token(s,271+i*46,379,32,27,'#91B76C');text(s,'Pᵗˣᵗ',259,414,92,28,23,C.green,true,'center');text(s,'text CE',260,447,90,20,16,C.gray,false,'center');
for(let i=0;i<3;i++)token(s,398+i*45,379,34,27,'#8BC4BF','Q');text(s,'Learned queries',379,412,170,28,20,C.teal,true,'center');text(s,'endpoint pose loss',379,447,170,20,16,C.gray,false,'center');
for(let i=0;i<6;i++)token(s,613+i*54,379,37,27,'#BBD1E8');text(s,'VAE + noisy keyframe tokens',572,412,420,28,21,C.blue,true,'center');text(s,'event roles + order + camera masks',572,447,420,20,16,C.gray,false,'center');
for(let i=0;i<5;i++)token(s,1051+i*54,379,37,27,'#D5BFE9');text(s,'State prefix + noisy actions',1018,412,320,28,21,C.violet,true,'center');text(s,'action flow loss',1018,447,320,20,16,C.gray,false,'center');
// bottom observed context and training / causal explanation
await img(s,45,28,487,123,99);text(s,'“Put the bowl\non the plate”',164,492,199,56,22,C.ink,true);text(s,'Current state sₜ',164,554,196,26,18,C.gray);
line(s,389,490,389,577,'#D7DFE4',1);
text(s,'One terminal pose',411,489,245,25,20,C.teal,true);text(s,'Query predicts position and orientation',411,523,350,24,17,C.ink);text(s,'Each primitive selects its meaningful events',411,552,350,24,16,C.gray);
line(s,785,490,785,577,'#D7DFE4',1);text(s,'Joint image / action denoising',809,489,510,25,20,C.ink,true);text(s,'Action tokens read Query, state and visual tokens',809,523,520,24,17,C.ink);text(s,'Frames illustrate the representation, not model outputs',809,552,520,24,16,C.gray);
s.speakerNotes.text='Design reference: user-provided aw0-framework-recreated(1).pptx, editable slide 2. The expert band, code example region, image output strip, token rows, and shared-attention band retain the reference composition. Removed pure/hybrid code-execution sidebar. AtomVLA content follows current sec/3_method.tex: the text head emits discrete names and string arguments, Query predicts one terminal pose per active arm, and future imagination substitutes for an executable function body. Image/action experts jointly denoise concatenated streams; the top frames do not imply staged image-only rollout or multiple Query waypoints. Visual examples and action curve derive from a local LIBERO source demonstration, not AtomVLA generated predictions. Pick illustrates pregrasp, closure, lift; move illustrates its endpoint. These demonstration frames illustrate event roles, not verified automated annotations or generated predictions. The number of events depends on the primitive and current phase, not a fixed uniform temporal grid.';
await save(p,'AtomVLA_method_pipeline');}
