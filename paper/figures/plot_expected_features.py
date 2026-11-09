"""Synthetic prospective visualization, NOT extracted model features.
16 semantic labels follow Physical Words Sec. 3.3. This plot is a hypothesis.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
out=Path(__file__).resolve().parent
rng=np.random.default_rng(20261102)
verbs=['Approach','Move','Pick','Place','Flip','Rotate','Toggle','Open','Close','Push','Pull','Insert','Press','Click','Strike','Handover']
n_per=400;dim=64;k=len(verbs);labels=np.repeat(np.arange(k),n_per)
basis=np.linalg.qr(rng.normal(size=(dim,dim)))[0]
centers=np.array([(x,y) for y in [-3,-1,1,3] for x in [-3,-1,1,3]],float)
centers+=rng.normal(scale=.19,size=(k,2))
# Unequal anisotropic clouds avoid duplicating panels or drawing identical blobs.
text=np.zeros((k*n_per,dim));pq=np.zeros_like(text)
for j in range(k):
 sl=slice(j*n_per,(j+1)*n_per)
 theta=rng.uniform(0,np.pi);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 txt_noise=rng.normal(size=(n_per,2))@np.diag([.74,.49])@rot.T
 pq_noise=rng.normal(size=(n_per,2))@np.diag([rng.uniform(.19,.31),rng.uniform(.12,.19)])@rot.T
 text[sl,:2]=.55*centers[j]+txt_noise
 pq[sl,:2]=centers[j]+pq_noise
 text[sl,2:]=rng.normal(scale=.32,size=(n_per,dim-2))
 pq[sl,2:]=rng.normal(scale=.14,size=(n_per,dim-2))
text=text@basis.T;pq=pq@basis.T
features=np.stack([text,pq]);flat=features.reshape(-1,dim)
centered=flat-flat.mean(axis=0);_,singular,comp=np.linalg.svd(centered,full_matrices=False)
coords=(centered@comp[:2].T).reshape(2,-1,2)
np.savez_compressed(out/'synthetic_features.npz',features=features,labels=labels,coordinates=coords,pca_components=comp[:2],feature_mean=flat.mean(axis=0))
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#6A3D9A','#A6761D','#E7298A','#4D4D4D','#1B9E77','#7570B3','#E31A1C','#66A61E','#B8860B','#174A6E']
markers=['o','s','^','D']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.8,6.3),sharex=True,sharey=True)
for v,(ax,title) in enumerate(zip(axes,['(a) Text','(b) Program + Query'])):
 order=rng.permutation(k)
 for j in order:
  pts=coords[v,labels==j]
  ax.scatter(pts[:,0],pts[:,1],s=5.5,c=colors[j],marker=markers[j%4],alpha=.52 if v==0 else .68,linewidths=0,rasterized=True)
  if v==1:
   c=pts.mean(axis=0);ax.text(c[0],c[1],str(j+1),fontsize=8,weight='bold',ha='center',va='center',bbox=dict(boxstyle='circle,pad=.12',facecolor='white',edgecolor=colors[j],linewidth=.6,alpha=.96))
 ax.set_title(title,fontweight='bold',pad=12)
 ax.set_xlabel('PC 1');ax.set_ylabel('PC 2' if v==0 else '')
 ax.set_aspect('equal',adjustable='box');ax.set_xticks([]);ax.set_yticks([])
 ax.set_facecolor('#FAFBFC');ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#CDD3DA')
lim=np.abs(coords).max()+.2
for ax in axes:ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)
handles=[Line2D([0],[0],marker=markers[j%4],linestyle='',color=colors[j],markersize=5,label=f'{j+1}. {v}') for j,v in enumerate(verbs)]
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.5,.025),handletextpad=.35,columnspacing=1.8)
fig.suptitle('SYNTHETIC EXPECTED FEATURES — NOT MEASURED MODEL OUTPUTS',fontsize=10,color='#8A3B12',fontweight='bold',y=.98)
fig.text(.5,.008,'16 action labels · 400 points per class per panel · 6,400 points per panel · shared PCA of 64-D synthetic features',ha='center',fontsize=8,color='#555555')
fig.subplots_adjust(left=.04,right=.98,top=.90,bottom=.25,wspace=.13)
for ext in ['png','pdf']:fig.savefig(out/f'expected_feature_comparison.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
(out/'figure_metadata.json').write_text(json.dumps({'status':'SYNTHETIC_PLANNING_ILLUSTRATION','reference':'Physical Words, arXiv:2610.26854, Sec. 3.3 vocabulary','classes':verbs,'samples_per_class_per_panel':400,'points_per_panel':6400,'total_points':12800,'dimensions':64,'seed':20261102,'projection':'joint PCA via SVD, centered without row normalization','explained_variance_ratio':(singular[:2]**2/(singular**2).sum()).tolist(),'panels':['Text','Program + Query'],'interpretation':'Program+Query separation is imposed by the synthetic generator. It is a revised expected outcome, not evidence and not the previous overlap hypothesis.'},indent=2))
print('Created two-panel synthetic feature comparison with 12,800 points.')
