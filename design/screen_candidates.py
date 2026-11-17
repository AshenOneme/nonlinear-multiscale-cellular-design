"""Reuse the frozen screened bank, paired native curves and exact-image elastic tensors.
No wall layout, nonlinear port model or cyclic-performance claim is produced here.
"""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument("--model-store",type=Path,required=True);a=ap.parse_args()
 r=a.project.resolve();o=a.out.resolve()
 bank=r/'data/cell_library/Ranked_Cell_Candidates_v1.h5'
 assert sha(bank)=='e9479ec25c34acb2b202f3c015e6053b1c063445d365bccfe59a9d3a2d3be1c8'
 root=a.model_store.resolve()/'Feng2025_full_library_macro8_v1'
 catp=root/'00_source_catalogue/all_original_records.json';tp=root/'02_elastic_tensor_bank/all_rows_tensors.npz';cp=tp.with_name('complete_immutable.json')
 complete=json.loads(cp.read_text(encoding='utf8'));assert complete['status']=='COMPLETE' and complete['source_identity']['E_normalized']==1 and complete['source_identity']['nu']==.3
 assert sha(tp)==complete['tensor_file_SHA']
 catalogue=json.loads(catp.read_text(encoding='utf8'));lookup={x['image_uint8_SHA']:x['row'] for x in catalogue}
 with np.load(tp) as z:C_all=z['C'].copy();qualified=z['qualified'].copy()
 with h5py.File(bank,'r') as f:
  g=f['train/original'];rows=g['source_row'][:];images=np.asarray(g['image'][:]).squeeze().astype(np.uint8)
  images=(images>0).astype(np.uint8);x=g['unit_TOP_X'][:];y=g['unit_TOP_Y'][:]
  ref=int(np.flatnonzero(rows==8624)[0]);counts=images.sum(axis=(1,2),dtype=np.int64)
  edge=np.concatenate([images[:,0,:],images[:,-1,:],images[:,:,0],images[:,:,-1]],axis=1)
  compatible=np.all(edge==edge[ref],axis=1)
  image_sha=np.array([hashlib.sha256(im.tobytes()).hexdigest() for im in images],dtype='S64')
  native_index=np.array([lookup.get(q.decode(),-1) for q in image_sha],dtype=np.int32)
  C=np.full((len(rows),3,3),np.nan);known=np.zeros(len(rows),dtype=bool)
  for i,j in enumerate(native_index):
   if j>=0 and qualified[j] and np.isfinite(C_all[j]).all() and np.min(np.linalg.eigvalsh(C_all[j]))>0:C[i]=C_all[j];known[i]=True
  for curve,key in [(x,'unit_TOP_X_sha256'),(y,'unit_TOP_Y_sha256')]:
   expected=g[key][:]
   assert all(hashlib.sha256(curve[i].tobytes()).hexdigest()==(s.decode() if isinstance(s,bytes) else s) for i,s in enumerate(expected))
 def integrate(curve):
  z=np.abs(curve);dis=z[:,:,0];force=z[:,:,1]
  assert np.isfinite(z).all() and np.all(dis[:,0]==0) and np.all(np.diff(dis,axis=1)>0)
  return np.c_[np.zeros(len(rows)),np.cumsum(.5*(force[:,1:]+force[:,:-1])*np.diff(dis,axis=1),axis=1)]
 wx,wy=integrate(x),integrate(y)
 o.mkdir(parents=True,exist_ok=True);p=o/'screened_candidate_features.npz';marker=o/'qualification_immutable.json'
 assert not marker.exists()
 np.savez_compressed(p,source_rows=rows,pixel_counts=counts,native_TOP_X=x,native_TOP_Y=y,
   monotonic_work_TOP_X=wx,monotonic_work_TOP_Y=wy,image_uint8_SHA=image_sha,
   baseline_four_boundary_signature_match=compatible,reference8624_index=np.array(ref),
   exact_image_join_native_index=native_index,periodic_elastic_C_E1_nu03=C,periodic_elastic_known=known)
 q={'schema':'screened_curve_energy_flow_candidate_features_v1','source_bank_SHA':sha(bank),'source_tensor_file_SHA':sha(tp),
  'source_tensor_qualification_SHA':sha(cp),'source_catalogue_SHA':sha(catp),'source_join':'exact_normalized_uint8_image_SHA_not_row_index',
  'candidate_group':'train/original; test/original not included or relabeled','candidate_count':len(rows),'reference_source_row':8624,
  'baseline_pixel_count':int(counts[ref]),'same_four_boundary_signature_count':int(compatible.sum()),'exact_image_elastic_tensor_qualified':int(known.sum()),
  'source_native_monotonic_curves_full_SHA_verified':True,'monotonic_work_not_cyclic_dissipation':True,'geometry_trace_match_not_full_interface_FE_qualification':True,
  'C_scope':'initial periodic plane-stress elasticity only, not nonlinear cyclic port operator','new_H_built':False,'new_FE_submitted':0,
  'files':[{'file':p.name,'SHA':sha(p),'bytes':p.stat().st_size}],'builder_SHA':sha(Path(__file__))}
 marker.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'candidate_count':len(rows),'reference_pixels':int(counts[ref]),'same_boundary_count':int(compatible.sum()),'matched_qualified_elastic_C':int(known.sum()),'proof_SHA':sha(marker),'new_H_built':False}))
if __name__=='__main__':main()
