"""Initial-elastic Q4 compatibility proxy; never a cyclic or 3D-wall predictor.
Dirichlet reaction stiffness is concave in element constitutive matrices.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
def quadrature(pitch=100.,thickness=5.):
 xy=np.array([[0,0],[pitch,0],[pitch,pitch],[0,pitch]],float)
 signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float);B=[];w=[]
 for xi in [-1/np.sqrt(3),1/np.sqrt(3)]:
  for eta in [-1/np.sqrt(3),1/np.sqrt(3)]:
   D=np.c_[signs[:,0]*(1+signs[:,1]*eta),signs[:,1]*(1+signs[:,0]*xi)]/4
   J=xy.T@D;grad=D@np.linalg.inv(J);b=np.zeros((3,8))
   b[0,0::2]=grad[:,0];b[1,1::2]=grad[:,1];b[2,0::2]=grad[:,1];b[2,1::2]=grad[:,0]
   B.append(b);w.append(np.linalg.det(J)*thickness)
 return np.array(B),np.array(w)
def mesh(mask):
 cells=np.argwhere(mask);corners=[]
 for rr,cc in cells:
  y=(len(mask)-rr-1)*100;x=cc*100
  corners.append([[x,y],[x+100,y],[x+100,y+100],[x,y+100]])
 xy,inv=np.unique(np.asarray(corners).reshape(-1,2),axis=0,return_inverse=True);conn=inv.reshape(-1,4)
 dof=(2*conn[:,:,None]+np.array([0,1])).reshape(-1,8)
 return cells,xy,conn,dof
def solve(mask,C):
 cells,xy,conn,dof=mesh(mask);n=len(xy)*2;B,w=quadrature()
 assert C.shape==(len(cells),3,3) and np.min(np.linalg.eigvalsh(C))>0
 ke=np.einsum('gai,eab,gbj,g->eij',B,C,B,w,optimize=True)
 K=coo_matrix((ke.ravel(),(np.repeat(dof,8,axis=1).ravel(),np.tile(dof,(1,8)).ravel())),shape=(n,n)).tocsr()
 top=np.flatnonzero(xy[:,1]==len(mask)*100);bottom=np.flatnonzero(xy[:,1]==0)
 fixed=np.unique(np.r_[2*top,2*top+1,2*bottom,2*bottom+1]);free=np.setdiff1d(np.arange(n),fixed)
 u=np.zeros(n);u[2*top]=1
 u[free]=spsolve(K[free][:,free],-K[free][:,fixed]@u[fixed])
 reaction=K@u;stiffness=float(reaction[2*top].sum());energy=float(u@reaction)
 assert stiffness>0 and abs(energy/stiffness-1)<1e-8
 assert np.max(abs(reaction[free]))/max(1.,stiffness)<1e-8
 strain=np.einsum('gai,ei->ega',B,u[dof])
 M=np.einsum('ega,egb,g->eab',strain,strain,w)
 assert abs(np.einsum('eab,eab->',M,C)/stiffness-1)<1e-8
 return dict(K=stiffness,element_gradient_C=M,cells=cells,xy=xy,conn=conn,nodal_u_unit=u.reshape(-1,2),
   energy_reaction_relative_error=abs(energy/stiffness-1),free_residual_relative=float(np.max(abs(reaction[free]))/stiffness) if len(free) else 0.)
def analytic_checks():
 E=206000.;nu=.3;C=E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
 B,w=quadrature();xy=np.array([[0,0],[100,0],[100,100],[0,100]],float)
 rigid=np.c_[-.01*xy[:,1]+3,.01*xy[:,0]-2].ravel()
 assert np.max(abs(B@rigid))<1e-15
 shear=np.c_[.01*xy[:,1],np.zeros(4)].ravel();strains=B@shear
 energy=np.einsum('ga,ab,gb,g->',strains,C,strains,w)
 expected=100*100*5*C[2,2]*.01**2
 assert abs(energy/expected-1)<1e-12
 return {'rigid_motion_zero_linear_strain':True,'constant_shear_exact_energy':True,
  'scope':'mathematical Q4 patch identities; not independent nonlinear Abaqus validation'}
