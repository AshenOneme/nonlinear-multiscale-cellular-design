"""Median-anchored energy/flow pixel proposal with an explicitly limited elastic proxy.
All physics-validation and cyclic-performance claims remain pending actual H FE.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,h5py
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix,vstack
from PIL import Image
import initial_compatibility as proxy
from material_motion import prepare,evaluate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,q):p.write_text(json.dumps(q,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def work(curve,cumulative,d):
 a=np.abs(curve);x=a[:,0];y=a[:,1]
 if d>x[-1]+1e-9:raise ValueError('Native curve range exceeded; no clipping or extrapolation')
 k=min(int(np.searchsorted(x,d,side='right')-1),len(x)-2);k=max(k,0)
 f=y[k]+(y[k+1]-y[k])*(d-x[k])/(x[k+1]-x[k])
 return float(cumulative[k]+.5*(y[k]+f)*(d-x[k]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prior-log',type=Path,required=True);ap.add_argument('--prior-log-sha',required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--demand',type=Path,required=True);ap.add_argument('--metrics',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument("--model-store",type=Path,required=True);a=ap.parse_args()
 r=a.project.resolve();o=a.out.resolve()
 o.mkdir(parents=True,exist_ok=True);assert not(o/'proposal_immutable.json').exists()
 demand=json.loads(a.demand.read_text(encoding='utf-8'));metrics=json.loads(a.metrics.read_text(encoding='utf-8-sig'))
 assert sha(a.metrics)==demand['source']['metrics_SHA']
 fp=a.metrics.parent/metrics['fields_file'];assert sha(fp)==metrics['fields_sha256']==demand['source']['fields_SHA']
 cache=a.model_store.resolve()/'Continuous_plate_macro8_M100_T1_v1/design_candidate_cache_v1'
 proof=json.loads((cache/'qualification_immutable.json').read_text());assert proof['candidate_count']==3271
 cp=cache/'screened_candidate_features.npz';assert sha(cp)==proof['files'][0]['SHA']
 z=np.load(cp);rows=z['source_rows'];pixels=z['pixel_counts'];ref=int(z['reference8624_index'])
 assert rows[ref]==8624 and pixels[ref]==26440
 rawx=z['native_TOP_X'];rawy=z['native_TOP_Y'];wx=z['monotonic_work_TOP_X'];wy=z['monotonic_work_TOP_Y']
 C=z['periodic_elastic_C_E1_nu03']*206000
 eligible=z['periodic_elastic_known']&z['baseline_four_boundary_signature_match']
 assert np.all(eligible)
 W2x=np.array([work(rawx[i],wx[i],2.) for i in range(len(rows))]);W2y=np.array([work(rawy[i],wy[i],2.) for i in range(len(rows))])
 eligible &= (W2x>0)&(W2y>0)
 bank=r/'data/cell_library/Ranked_Cell_Candidates_v1.h5';assert sha(bank)==proof['source_bank_SHA']
 fields=np.load(fp);mask=fields['macro_active'];nodes=fields['reference_nodes_xyz_mm'];conn=fields['element_connectivity_zero_based'];slots=fields['cell_slots']
 cells,xy,plane_conn,dof=proxy.mesh(mask);lookup={(int(rr),int(cc)):i for i,(rr,cc) in enumerate(cells)}
 demand_by={(q['row'],q['col']):q for q in demand['cells']}
 symmetric=bool(np.array_equal(mask,mask[:,::-1]));groups=[];used=set()
 for rr,cc in cells:
  rc=(int(rr),int(cc))
  if rc in used:continue
  members=[rc]
  other=(int(rr),int(mask.shape[1]-1-cc))
  if symmetric and other!=rc:members.append(other)
  assert all(q in lookup for q in members)
  used.update(members);groups.append(members)
 mult=np.array([len(g) for g in groups]);ns=len(groups)
 energy=np.array([np.mean([demand_by[q]['plastic_dissipation_full_nominal2_cycle_Nmm'] for q in g]) for g in groups])
 positive=energy[energy>max(1.,energy.max())*1e-7];assert len(positive)>0
 median=float(np.median(positive));target=energy/median
 betas=[];direction_proxy_fallback=[]
 for group in groups:
  samples=[s for rc in group for s in demand_by[rc]['signed_states'].values()]
  outside=any(abs(s['engineering_shear_boundary_amplitude_mm'])>rawx[ref,-1,0]+1e-9 or s['compression_y_amplitude_mm']>abs(rawy[ref,-1,0])+1e-9 for s in samples)
  direction_proxy_fallback.append(outside);sx=sy=0.
  for s in samples:
   dx=abs(s['engineering_shear_boundary_amplitude_mm']);dy=s['compression_y_amplitude_mm']
   if outside:
    # Only a direction-mixture weight. Actual deformation is retained, not
    # clipped; no native curve outside 0..5 mm is evaluated or extrapolated.
    sx+=C[ref,2,2]*(dx/100)**2
    sy+=(C[ref,1,1]-C[ref,0,1]**2/C[ref,0,0])*(dy/100)**2
   else:
    sx+=work(rawx[ref],wx[ref],dx);sy+=work(rawy[ref],wy[ref],dy)
  betas.append(sx/(sx+sy) if sx+sy>0 else .5)
 beta=np.array(betas);cap=beta[:,None]*W2x+(1-beta[:,None])*W2y
 cap_ref=beta*W2x[ref]+(1-beta)*W2y[ref]
 ratio=cap/cap_ref[:,None]
 cost=(np.arcsinh(ratio-1)-np.arcsinh(target[:,None]-1))**2
 cost*=mult[:,None]
 # True signed/IP tensors enter a quadratic stress-shape diagnostic, never a signed average.
 A=np.zeros((len(cells),3,3))
 for timevalue in [13,15]:
  U=fields['time_%02d_U_xyz_mm'%timevalue];sv=fields['time_%02d_S'%timevalue].reshape(len(conn),8,6)
  for e,(rr,cc,layer) in enumerate(slots):
   fit=evaluate(U[conn[e]],prepare(nodes[conn[e]],np.ones(8)))
   F=np.array(fit['fitted_F_xyz']);u,s,vt=np.linalg.svd(F);R=u@vt;assert np.linalg.det(R)>0
   for v in sv[e]:
    sigma=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
    co=R.T@sigma@R;v2=np.array([co[0,0],co[1,1],co[0,1]])
    A[lookup[(int(rr),int(cc))]]+=np.outer(v2,v2)/16
 T=np.diag([1,1,-1]);Cmirror=T[None]@C@T[None];S=np.linalg.inv(C);Smirror=np.linalg.inv(Cmirror)
 flow=np.zeros((ns,len(rows)))
 for k,group in enumerate(groups):
  for which,rc in enumerate(group):
   Ae=A[lookup[rc]];Suse=Smirror if symmetric and which==1 else S
   base=float(np.sum(Ae*(Smirror[ref] if symmetric and which==1 else S[ref])))
   if base>0:flow[k]+=np.einsum('ab,jab->j',Ae,Suse)/base
 # Dictionary: mechanics-specific resource-support cases plus deterministic energy-context coverage.
 palette=[ref]
 def add(i):
  i=int(i)
  if eligible[i] and i not in palette:palette.append(i)
 for score in [C[:,2,2]/pixels,(C[:,0,0]-C[:,0,1]**2/C[:,1,1])/pixels,W2x/pixels,W2y/pixels,-pixels.astype(float)]:
  scores=np.where(eligible,score,-np.inf);add(np.argmax(scores))
 while len(palette)<20:
  current=np.min(cost[:,palette],axis=1)
  gain=np.maximum(0,current[:,None]-cost).sum(axis=0);gain[~eligible]=-np.inf;gain[palette]=-np.inf
  best=float(np.max(gain))
  if best<1e-12:break
  options=np.flatnonzero(abs(gain-best)<=1e-12)
  order=sorted(options,key=lambda j:(float(flow[:,j].sum()),bytes(z['image_uint8_SHA'][j])))
  add(order[0])
 palette=np.array(palette,dtype=int);nc=len(palette);Cdict=C[palette];Cdict_m=Cmirror[palette]
 site_of=np.zeros(len(cells),int);mirror_of=np.zeros(len(cells),bool)
 for k,g in enumerate(groups):
  for which,rc in enumerate(g):site_of[lookup[rc]]=k;mirror_of[lookup[rc]]=symmetric and which==1
 base=proxy.solve(mask,np.repeat(C[ref][None],len(cells),axis=0));Kbase=base['K']
 registration={'method':'energy_assignment_same_inventory_shared_equilibrium_swap_repair_v5',
  'source_demand_SHA':sha(a.demand),'source_metrics_SHA':sha(a.metrics),'source_fields_SHA':sha(fp),
  'candidate_proof_SHA':sha(cache/'qualification_immutable.json'),'selector_SHA':sha(Path(__file__)),
  'initial_proxy_module_SHA':sha(Path(proxy.__file__)),'before_any_new_H_FE':True,
  'canonical_unit_capacity_probe_mm':2,'capacity_measure':'original native monotonic loading work only',
  'energy_target':'paired actual full2%cycle ELPD increment / median positive paired increments; baseline8624 anchored',
  'assignment_metric':'squared asinh(capacity_ratio-1) distance to asinh(demand_ratio-1); no field rescaling',
  'direction_mix':'native work fractions within measured curve domain; whole-group initial-elastic kinematic modal weight only when any native test domain is exceeded',
  'direction_mix_elastic_fallback_groups':int(sum(direction_proxy_fallback)),
  'out_of_range_deformation_preserved_no_curve_extrapolation_or_clipping':True,
  'fallback_is_not_actual_cyclic_energy_partition':True,
  'maximum_dictionary_types':20,'force_minimum_type_count':False,'median_anchor_group':int(np.argmin(abs(target-1))),
  'pixel_budget':'exact equality to corresponding original8624; also <=min(Q02,1.20all9481)',
  'initial_compatibility_constraint':'K_proxy(layout)>=K_proxy(all8624); concave Dirichlet energy outer-approximation',
  'initial_proxy_scope':'periodic linear elastic Q4 only; not a nonlinear port or qualified true wall response',
  'D03_known_initial_proxy_baseline_RF_error_relative_to_true_U':-.2042265146663853 if demand['macro_case']=='D03_wide_door' else None,
  'no_RF_or_stress_calibration_applied':True,'new_H_true_initial_and_cyclic_FE_required':True,
  'lexicographic_flow_step':'flow diagnostic/tie-break retained; no secondary global optimum claimed',
  'source_prior_v2_log_SHA':a.prior_log_sha,'original_type_inventory_and_pixel_budget_preserved':True,
  'physical_constraints_and_energy_fit_objective_unchanged':True,'actual_H_FE_results_used':False,
  'solver_time_limit_per_call_seconds':20,'maximum_repair_swaps':40,'maximum_energy_swaps':80,'maximum_actual_equilibrium_trials_per_swap':20,'physical_advantage_claimed':False}
 save(o/'registration_before_assignment.json',registration)
 patch=proxy.analytic_checks();anchor=registration['median_anchor_group'];nvar=ns*nc
 baseA=csr_matrix((np.ones(nvar),(np.repeat(np.arange(ns),nc),np.arange(nvar))),shape=(ns,nvar))
 pixrow=(mult[:,None]*pixels[palette][None]/pixels[ref]).reshape(1,-1)
 matrices=[baseA,csr_matrix(pixrow)];lower=[*np.ones(ns),float(len(cells))];upper=lower.copy()
 lo=np.zeros(nvar);hi=np.ones(nvar);hi[anchor*nc:(anchor+1)*nc]=0;hi[anchor*nc]=1
 bounds=Bounds(lo,hi);records=[]
 def C_layout(indices):
  cc=Cdict[indices[site_of]].copy();cc[mirror_of]=Cdict_m[indices[site_of[mirror_of]]];return cc
 def cut(solution):
  gradient=np.zeros((ns,nc));M=solution['element_gradient_C']
  for e in range(len(cells)):
   gradient[site_of[e]]+=np.einsum('ab,jab->j',M[e],Cdict_m if mirror_of[e] else Cdict)
  return csr_matrix((gradient/Kbase).reshape(1,-1))
 matrices.append(cut(base));lower.append(1.);upper.append(np.inf)
 def optimize(objective,phase):
  for it in range(48):
   Aall=vstack(matrices).tocsr()
   result=milp(objective,integrality=np.ones(nvar),bounds=bounds,
    constraints=LinearConstraint(Aall,np.array(lower),np.array(upper)),
    options={'time_limit':20.,'mip_rel_gap':1e-6})
   if result.x is None:raise RuntimeError('No feasible incumbent; no H fabricated: '+str(result.message))
   X=result.x.reshape(ns,nc);indices=X.argmax(axis=1);assert np.max(abs(X-np.eye(nc)[indices]))<1e-5
   material=int(np.sum(mult*pixels[palette[indices]]));assert material==int(len(cells)*pixels[ref])
   equilibrium=proxy.solve(mask,C_layout(indices));relative=equilibrium['K']/Kbase
   record={'phase':phase,'iteration':it,'milp_status':int(result.status),'mip_gap':float(getattr(result,'mip_gap',0)),
    'energy_fit':float(np.sum(cost[:,palette]*np.eye(nc)[indices])),'K_proxy_ratio':relative,'types':len(set(indices.tolist())),
    'material_pixels':material,'actual_microstructural_FE_performed':False,'selected_palette_indices':indices.tolist()}
   records.append(record);print(json.dumps(record),flush=True);save(o/'assignment_iteration_log.json',records)
   if relative>=1-1e-7:return indices,equilibrium,result
   matrices.append(cut(equilibrium));lower.append(1.);upper.append(np.inf)
  raise RuntimeError('Initial compatibility outer approximation unresolved; no unqualified H submission')
 assert sha(a.prior_log)==a.prior_log_sha
 prior=json.loads(a.prior_log.read_text(encoding='utf-8'));candidate=max((q for q in prior if q['phase']=='energy'),key=lambda q:q['K_proxy_ratio'])
 indices=np.array(candidate['selected_palette_indices'],dtype=int);assert len(indices)==ns and indices[anchor]==0 and np.all((indices>=0)&(indices<nc))
 inventory=np.bincount(indices,weights=mult,minlength=nc).astype(int);eobj=cost[:,palette];fobj=flow[:,palette]
 def assess(idx):
  assert idx[anchor]==0 and np.array_equal(np.bincount(idx,weights=mult,minlength=nc).astype(int),inventory)
  assert int(np.sum(mult*pixels[palette[idx]]))==len(cells)*int(pixels[ref])
  eq=proxy.solve(mask,C_layout(idx));fit=float(eobj[np.arange(ns),idx].sum());return eq,eq['K']/Kbase,fit
 equilibrium,K,fit=assess(indices);assert abs(K-candidate['K_proxy_ratio'])<1e-9 and abs(fit-candidate['energy_fit'])<1e-9
 records=[]
 def record(phase,it):
  q={'phase':phase,'iteration':it,'K_proxy_ratio':K,'energy_fit':fit,'types':len(set(indices.tolist())),'selected_palette_indices':indices.tolist(),'same_inventory_verified':True,'material_pixels':int(len(cells)*pixels[ref]),'actual_H_FE_performed':False};records.append(q);save(o/'assignment_iteration_log.json',records);print(json.dumps({k:v for k,v in q.items() if k!='selected_palette_indices'}),flush=True)
 record('replayed_start',0)
 pairs=[(i,j) for i in range(ns) for j in range(i+1,ns) if i!=anchor and j!=anchor and mult[i]==mult[j]]
 for it in range(40):
  if K>=1-1e-7:break
  gradient=cut(equilibrium).toarray().reshape(ns,nc);candidates=[]
  for i,j in pairs:
   a,b=indices[i],indices[j]
   if a==b:continue
   gain=gradient[i,b]+gradient[j,a]-gradient[i,a]-gradient[j,b]
   if gain>1e-9:candidates.append((-gain,float(eobj[i,b]+eobj[j,a]-eobj[i,a]-eobj[j,b]),i,j))
  candidates.sort();best=None
  for _,de,i,j in candidates[:20]:
   idx=indices.copy();idx[i],idx[j]=idx[j],idx[i];eq,kk,ff=assess(idx)
   if kk>K+1e-9 and (best is None or kk>best[2]+1e-12 or (abs(kk-best[2])<1e-12 and ff<best[3])):best=(idx,eq,kk,ff)
  if best is None:break
  indices,equilibrium,K,fit=best;record('proxy_stiffness_inventory_repair',it)
 assert K>=1-1e-7,'Inventory repair unresolved; original constraints not relaxed'
 for it in range(80):
  gradient=cut(equilibrium).toarray().reshape(ns,nc);candidates=[]
  for i,j in pairs:
   a,b=indices[i],indices[j]
   if a==b:continue
   delta=float(eobj[i,b]+eobj[j,a]-eobj[i,a]-eobj[j,b]);gain=gradient[i,b]+gradient[j,a]-gradient[i,a]-gradient[j,b]
   if delta<-1e-9 and K+gain>=1-1e-7:candidates.append((delta,float(fobj[i,b]+fobj[j,a]-fobj[i,a]-fobj[j,b]),i,j))
  candidates.sort();best=None
  for delta,df,i,j in candidates[:20]:
   idx=indices.copy();idx[i],idx[j]=idx[j],idx[i];eq,kk,ff=assess(idx)
   if kk>=1-1e-7 and ff<fit-1e-9:best=(idx,eq,kk,ff);break
  if best is None:break
  indices,equilibrium,K,fit=best;record('energy_improvement_same_inventory_and_feasible_K',it)
 assert fit<float(cost[:,ref].sum()),'No improvement over uniform energy-fit proxy'
 record('final_local_solution_no_global_optimum_claim',0)
 selected=palette[indices];matrix=np.full(mask.shape,-1,dtype=np.int32);reflect=np.zeros(mask.shape,bool)
 for k,g in enumerate(groups):
  for which,(rr,cc) in enumerate(g):matrix[rr,cc]=rows[selected[k]];reflect[rr,cc]=symmetric and which==1
 with h5py.File(bank,'r') as f:images=(np.asarray(f['train/original/image'][:]).squeeze()>0).astype(np.uint8)
 raster=np.zeros((mask.shape[0]*256,mask.shape[1]*256),np.uint8)
 for k,g in enumerate(groups):
  image=images[selected[k]]
  assert hashlib.sha256(image.tobytes()).hexdigest()==z['image_uint8_SHA'][selected[k]].decode()
  for rr,cc in g:raster[rr*256:(rr+1)*256,cc*256:(cc+1)*256]=image[:,::-1] if reflect[rr,cc] else image
 assert int(raster.sum())==int(len(cells)*pixels[ref])
 if symmetric:assert np.array_equal(raster,raster[:,::-1])
 uq,ct=np.unique(matrix[matrix>=0],return_counts=True)
 isH=len(uq)>1
 Image.fromarray(255*(1-raster)).save(o/'H_original_pixels.png')
 Image.fromarray(255*(1-raster)).resize((2400,1200),Image.Resampling.NEAREST).save(o/'H_pixel_preview.png')
 np.savez_compressed(o/'pixel_assignment.npz',source_rows=matrix,reflect_x=reflect,active=mask,original_binary_material=raster)
 ratio9481=float(pixels[ref]/pixels[int(np.flatnonzero(rows==9481)[0])]);assert ratio9481<=1.20+1e-12
 result={'macro_case':demand['macro_case'],'job_source':demand['job'],'is_heterogeneous_pixel_proposal':isH,
  'is_mesh_or_FE_qualified':False,'new_FE_submitted':0,'source':registration,'registration_SHA':sha(o/'registration_before_assignment.json'),
  'h5_rows':matrix.tolist(),'reflect_x':reflect.tolist(),'type_count':len(uq),'type_counts':{str(k):int(v) for k,v in zip(uq,ct)},
  'largest_type_fraction':float(ct.max()/ct.sum()),'pixel_material_count':int(raster.sum()),'Q02_pixel_material_count':int(len(cells)*pixels[ref]),
  'pixel_material_change_vs_Q02_percent':0.,'pixel_volume_fraction_effective_grid':float(pixels[ref]/65536),
  'reference8624_median_anchor_members':groups[anchor],'symmetric_macro':symmetric,'exact_pixel_mirror_verified':symmetric,
  'initial_proxy_K_N_per_mm':equilibrium['K'],'baseline_initial_proxy_K_N_per_mm':Kbase,
  'initial_proxy_K_ratio':equilibrium['K']/Kbase,'proxy_initial_identity_checks':patch,'solver_iterations':records,
  'material_mm3':float(raster.sum()*(100/256)**2*5),
  'evidence_limits':['No H actual RF/stress/energy result exists','Same pixel traces do not establish full mesh interface qualification',
   'Periodic elastic whole-wall proxy has known baseline error and is not a nonlinear prediction','Full actual shared-node FE re-equilibrium and cyclic feedback remain required']}
 result['outputs']=[{'file':p.name,'SHA':sha(p),'bytes':p.stat().st_size} for p in o.iterdir() if p.is_file()]
 save(o/'proposal_immutable.json',result)
 print(json.dumps({'proposal_created':isH,'type_count':len(uq),'max_type_fraction':result['largest_type_fraction'],
  'pixel_change_Q02_percent':0,'initial_proxy_ratio':equilibrium['K']/Kbase,'actual_H_FE_submitted':0,'proposal_SHA':sha(o/'proposal_immutable.json')}))
if __name__=='__main__':main()
