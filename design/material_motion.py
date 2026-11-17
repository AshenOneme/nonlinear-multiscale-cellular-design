"""Reference-material lumped nodal motion diagnostics, not energy or damage."""
import numpy as np


def cell_weights(conn,cell_slots,volume,n_cells=72):
    cell_ids=cell_slots[:,0]*12+cell_slots[:,1]
    keys,inverse=np.unique((conn*n_cells+cell_ids[:,None]).ravel(),return_inverse=True)
    values=np.zeros(len(keys))
    np.add.at(values,inverse,np.repeat(volume/6,6))
    out={}
    for cell in np.unique(cell_ids):
        active=keys%n_cells==cell
        ids=keys[active]//n_cells;w=values[active]
        assert abs(w.sum()-volume[cell_ids==cell].sum())<max(1e-7,w.sum()*1e-12)
        out[int(cell)]=(ids,w)
    return out


def prepare(x,w):
    assert x.shape==(len(w),3) and np.all(w>0) and np.isfinite(x).all()
    w=w/w.sum();center=w@x;z=x-center
    scale=np.sqrt(w@(z*z));assert np.all(scale>0)
    a=z/scale;cov=a.T@(w[:,None]*a)
    condition=float(np.linalg.cond(cov))
    if not np.isfinite(condition) or condition>1e10:raise ValueError('Singular material-node affine fit')
    return w,a,scale,np.linalg.inv(cov),condition


def evaluate(u,reference):
    w,a,scale,inv,condition=reference
    centered=u-w@u
    grad_scaled=centered.T@(w[:,None]*a)@inv
    grad=grad_scaled/scale[None,:]
    residual=centered-a@grad_scaled.T
    norms=np.linalg.norm(residual,axis=1)
    f=np.eye(3)+grad
    return {'reference_material_lumped_node_RMS_nonaffine_mm':float(np.sqrt(w@(norms*norms))),
            'maximum_material_node_nonaffine_mm':float(norms.max()),
            'fitted_displacement_gradient_xyz':grad.tolist(),'fitted_F_xyz':f.tolist(),
            'determinant_of_fitted_F_not_element_J':float(np.linalg.det(f)),
            'reference_fit_condition':condition}


def analytic_checks():
    x=np.array([[a,b,c] for c in (-2.5,2.5) for a,b in ((0,0),(100,0),(100,100),(0,100))],float)
    w=np.ones(len(x));ref=prepare(x,w)
    g=np.array([[.02,.03,0],[-.01,.04,.02],[0,.01,-.005]])
    u=x@g.T+[3.,-2.,1.];q=evaluate(u,ref)
    assert q['reference_material_lumped_node_RMS_nonaffine_mm']<1e-12
    assert np.max(abs(np.array(q['fitted_displacement_gradient_xyz'])-g))<1e-12
    non=u.copy();non[:,0]+=(x[:,0]-50)*(x[:,1]-50)/2500
    before=evaluate(non,ref)
    theta=.37;rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    transformed=(x+non)@rot.T+[8.,-1.,4.]-x
    after=evaluate(transformed,ref)
    assert abs(before['reference_material_lumped_node_RMS_nonaffine_mm']-after['reference_material_lumped_node_RMS_nonaffine_mm'])<1e-12
    assert before['reference_material_lumped_node_RMS_nonaffine_mm']>.9
    assert np.max(abs(np.array(after['fitted_F_xyz'])-rot@np.array(before['fitted_F_xyz'])))<1e-12
    return {'exact_affine_motion_zero_residual':True,'three_dimensional_superposed_rigid_rotation_objective':True,
            'higher_order_mode_remains_visible':True,'residual_is_not_strain_energy_or_failure':True}
