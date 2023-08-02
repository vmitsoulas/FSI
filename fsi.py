# P2-P0-P1 Fully Coupled FSI for: Hron - Turek benchmark (DOI: 10.1007/3-540-34596-5_15)
# Solution is stored in HDF5 format to favor further post process activities using Fenics.
from dolfin import *
import time

parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters["form_compiler"]["quadrature_degree"] = 3
parameters['form_compiler']["cpp_optimize_flags"] = '-O2 -funroll-loops'
PETScOptions.set("mat_mumps_icntl_14",40.0)
set_log_level(40)

# Global parameters
# Geometry quantities in m as defined in the original work of Hron & Turek
d  = 0.1
H  = 0.41
h  = 0.02
Ll = 2.5
ll = 0.35
cx = 0.2
cy = cx
hx = 0.6
hy = 0.19

# Characteristic velocity
u_av = 2.0

# Time control
dt = 0.0005 			# s
k  = Constant(dt*u_av/d)
T  = 8.0*u_av/d

# fluid material properties
rho_f = 1.0e3 			# kg/m3
nu_f  = 1.0e-3			# m2/s
mu_f  = nu_f*rho_f 		# Pa.s

# solid material properties
rho_s = 1.0e3 			# kg/m3
mu_s  = 2.0e6 			# Pa
nu_s  = 0.4 			# -
E     = 2.0*mu_s*(1.0+nu_s)
lamda = (nu_s*E)/((1.0+nu_s)*(1.0-2.0*nu_s))
AE    = E/(rho_f*u_av**2)

#Dimensionless Numbers
Re_f = d*rho_f*u_av/mu_f
if MPI.rank(MPI.comm_world) == 0 :
	print("reynolds number: ", Re_f)

Re_s = u_av**2*rho_f/mu_s
if MPI.rank(MPI.comm_world) == 0 :
	print("aeroelasticity number: ", AE)

Rho  = rho_s / rho_f
lm   = lamda/mu_s

# Penalty Parameters
delta  = Constant(1.0e+6)

# Mesh import
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(),"meshes/file.h5","r")
hdf.read(mesh,"/mesh",False)
subdomains = MeshFunction("size_t",mesh,2)
hdf.read(subdomains,"/subdomains")
boundaries = MeshFunction("size_t",mesh,1)
hdf.read(boundaries,"/boundaries")

def BroydenMethod(Residual,Jac,BC,BCh,uknownv,corr):
	
	a_tol , r_tol 	= 1.0e-8, 1.0e-8
	nIter 			= 0
	rel_res 		= 1.0
	fnorm 			= 1.0
	lmbda 			= 1.0
	[bc.apply(uknownv.vector()) for bc in BC]
	A 				= assemble(Jac)
	[bc.apply(A) for bc in BCh]

	
	while (rel_res >= r_tol and fnorm >= a_tol and nIter <= 20) :

		b 	= assemble(-Residual)
		solver = LUSolver(A,"mumps")		
		[bc.apply(b) for bc in BCh]
		solver.solve(A,corr.vector(),b)

		rel_res 	= corr.vector().norm('l2')
		fnorm 		= b.norm('l2')
		
		uknownv.vector()[:] += lmbda*corr.vector()[:]
		nIter 				+= 1
		A_PRE = A

	if MPI.rank(MPI.comm_world) == 0 :
		print( '{0:2d} {1:3.2E} {2:3.2E}'.format(nIter,rel_res,fnorm))

def Update_Displacement(MeshMove,MeshVel,MeshDisp,Timestep):
	MeshMove.assign(MeshVel)
	MeshMove.vector()[:] *= float(Timestep)
	MeshDisp.vector()[:] += MeshMove.vector()[:]
    
#_____________Boundaries_&_Subdomains_ID____________________
Top 		= 2
Bottom 		= 3
inlet 		= 1
solidbase 	= 6
outlet 		= 4
solid 		= 8
fluid 		= 9
interface	= 7
hole 		= 5

#_______________FunctionSpaces__________________
V = VectorElement('CG',mesh.ufl_cell(),2)
P = FiniteElement('DG',mesh.ufl_cell(),0)
W = VectorElement('CG',mesh.ufl_cell(),1)

VPW = FunctionSpace(mesh,MixedElement([V,P,W]))

# Test Functions
phi,eta,psi = TestFunctions(VPW)


#_____________Boundary_Conditions_______________________________
noSlipNoPen  = Constant((0.0,0.0))
noSlip = Constant(0.0)
noPen = Constant(0.0)
Pref = Constant(0.0)
V_in = ('(u_av*1.5*x[1]*(H/d - x[1]) / pow(H/(2*d), 2))/u_av','0')
V_in_t = ('((u_av*1.5*x[1]*(H/d - x[1]) / pow(H/(2*d), 2))*0.5*(1-cos(pi*0.5*t)))/u_av','0')
V_int = Expression(V_in_t,t=0,u_av=u_av,H=H,d=d,degree=2)

# Top
bctop1   =  DirichletBC(VPW.sub(0),noSlipNoPen,boundaries,Top)
bctop2   =  DirichletBC(VPW.sub(2).sub(1),noPen,boundaries,Top)
# Bottom
bcbot1   =  DirichletBC(VPW.sub(0),noSlipNoPen,boundaries,Bottom)
bcbot2   =  DirichletBC(VPW.sub(2).sub(1),noPen,boundaries,Bottom)
# Inlet
bcin1    =  DirichletBC(VPW.sub(0),Expression(V_in,u_av=u_av,H=H,d=d,degree=2),boundaries,inlet)
bcin1_t  =  DirichletBC(VPW.sub(0),V_int,boundaries,inlet)
bcin2    =  DirichletBC(VPW.sub(2).sub(0),noPen,boundaries,inlet)
bcinh    =  DirichletBC(VPW.sub(0),noSlipNoPen,boundaries,inlet)
# Elastic beam base 
bcfixed1 =  DirichletBC(VPW.sub(2),noSlipNoPen,boundaries,solidbase)
bcfixed2 =  DirichletBC(VPW.sub(0),noSlipNoPen,boundaries,solidbase)
# Cylinder
bchole   =  DirichletBC(VPW.sub(2),noSlipNoPen,boundaries,hole)
bchole2  =  DirichletBC(VPW.sub(0),noSlipNoPen,boundaries,hole)
# Outlet
bcout2   =  DirichletBC(VPW.sub(2).sub(0),noPen,boundaries,outlet)
bcpout 	 =  DirichletBC(VPW.sub(1),Pref,boundaries,outlet)

# BC Lists
bcs_du = [bchole,bchole2,bctop1,bctop2,bcbot1,bcbot2,bcin2,bcinh,bcfixed1,bcfixed2,bcout2,bcpout]
bcs    = [bchole,bchole2,bcpout,bcout2,bctop1,bctop2,bcbot1,bcbot2,bcin1,bcin2,bcfixed1,bcfixed2]
bcs_t  = [bchole,bchole2,bcpout,bcout2,bctop1,bctop2,bcbot1,bcbot2,bcin1_t,bcin2,bcfixed1,bcfixed2]

#___________Integration_Domains______________________________
dS = Measure("dS")(subdomain_data=boundaries)
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dx_f = dx(fluid,subdomain_data=subdomains)
dx_s = dx(solid,subdomain_data=subdomains)


# Useful python functions for stress tensor definitions used in the weak forms
def F(Vel,Dis,Timestep) :
	return Identity(len(Vel)) + grad(Vel*Timestep + Dis)

def C(Vel,Dis,Timestep) :
	return F(Vel,Dis,Timestep).T * F(Vel,Dis,Timestep)

def EPS(Vel,Dis,Timestep):
    return 0.5*(C(Vel,Dis,Timestep)-Identity(len(Vel)))

def eps(xx):
	return sym(grad(xx))

def Piola_2nd(Vel,Dis,Timestep,Press):
	return 2.0*EPS(Vel,Dis,Timestep) - Press*Identity(len(Vel))

def sigma_mesh(xx):
	return 2.0*eps(xx) + lm*tr(eps(xx))*Identity(len(xx))

def sigma_f(Vel, Press):
	return 2.0*eps(Vel) - Press*Identity(len(Vel))

def Piola_1st(Vel,Dis,Timestep,Press):
	return F(Vel,Dis,Timestep)*Piola_2nd(Vel,Dis,Timestep,Press)

def J(Vel,Dis,Timestep) :
	return det(F(Vel,Dis,Timestep))

def sigma_def(mesh_Vel,Vel,Press,Dis,Timestep) :
	return (grad(Vel)*inv(F(mesh_Vel,Dis,Timestep)) + inv(F(mesh_Vel,Dis,Timestep)).T*(grad(Vel).T) - Press*Identity(len(Vel)))

# UFL Functions to store unknowns in previous time steps
U  	= Function(VPW.sub(2).collapse())
v0 	= Function(VPW.sub(0).collapse())
# Dummy UFL function
w_ 	= Function(VPW.sub(2).collapse())


# Unknown vector
unknown = Function(VPW)
v,p,w = split(unknown)


########################## WEAK FORMS IN DIMENSIONLESS FORM #################################
##### FLUID DOMAIN
# ALE momentum in the undeformed configuration
aMF			= ( 1.0/k * inner( J(w,U,k)*(v-v0) , phi ) \
	        +   inner( dot( J(w,U,k)*grad(v) , dot( inv(F(w,U,k)) , (v-w) ) ) , phi )  \
	        + 1.0/Re_f * inner(J(w,U,k)*sigma_def(w,v,p,U,k)*inv(F(w,U,k)).T , grad(phi) ) )*dx_f
	
# Continuity in the undeformed configuration	
aCF 		= inner(div(J(w,U,k)*dot(inv(F(w,U,k)),v)) , eta)*dx_f

# Equation of mesh motion in the undeformed configuration - Laplace of the displacement
aDF 	      = - (k*inner( 1.0/J(w,U,k)*grad(w) , grad(psi) ) \
              +    inner( 1.0/J(w,U,k)*grad(U) , grad(psi) ))*dx_f								

aF = aMF + aCF + aDF 
#####

##### SOLID DOMAIN
# Momentum
aMS      = (Rho*(1.0/k * inner( (v-v0) , phi ))                            \
	     + 1.0/Re_s * inner(  Piola_1st(v,U,k,p) , grad(phi) ))*dx_s

# Mesh motion pseudo-equation
aDS      = delta*inner(v,psi)*dx_s \
	     - delta*inner(w,psi)*dx_s

# Incompressibility
aCS 	=  (lm*tr(EPS(v,U,k))*eta)*dx_s

aS  = aMS + aDS + aCS
#####
a   = aS  + aF
########################################################################################


# RES & JAC
R           = a
correction  = Function(VPW)
DR          = derivative(R,unknown)#, trial)

# Loop Control Parameters
t           = 0.0
max_iter    = 100
max_er      = 1e-4
ite         = 0
Time_interval = 20

timer0 = time.process_time()
while t < 2.0*u_av/d :

	BroydenMethod(R,DR,bcs_t,bcs_du,unknown,correction)
	v,p,w = unknown.split(True)
	# ALE mesh displacement
	Update_Displacement(w_,w,U,k)

	if ite % Time_interval == 0 :
		NO = (ite/Time_interval)
		VFile 		= HDF5File(mesh.mpi_comm(),"solution/velocity/u%s.h5" %NO,"w")
		DFile 		= HDF5File(mesh.mpi_comm(),"solution/displacement/d%s.h5"%NO,"w")
		PFile 		= HDF5File(mesh.mpi_comm(),"solution/pressure/p%s.h5"%NO,"w")
		V0File 		= HDF5File(mesh.mpi_comm(),"solution/velocity0/u0%s.h5"%NO,"w")
		WFile 		= HDF5File(mesh.mpi_comm(),"solution/meshvel/w%s.h5"%NO,"w")

		VFile.write(v,"vel" ,ite)
		DFile.write(U,"dis" ,ite)
		PFile.write(p,"pre" ,ite)
		V0File.write(v0,"vel0" ,ite)
		WFile.write(w,"mesh_vel" ,ite)
		VFile.close()
		PFile.close()
		DFile.close()
		V0File.close()
		WFile.close()
		if MPI.rank(MPI.comm_world) == 0 :
			print ('time = ', (t*d/u_av))
			print ('ite_no = ' , (ite))

	# Update previous step velocity
	v0.assign(v)
	t += float(k)
	V_int.t = t*d/u_av
	ite += 1

print("###############################################################################")
print("################                                                ###############")
print("################ Fully developed inlet velocity profile reached ###############")
print("################                                                ###############")
print("###############################################################################")

for bc in bcs :
	bc.apply(unknown.vector())
while t >= 2.0*u_av/d and t < T :
	
	BroydenMethod(R,DR,bcs,bcs_du,unknown,correction)
	v,p,w = unknown.split(True)
	# ALE mesh displacement
	Update_Displacement(w_,w,U,k)

	if ite % Time_interval == 0 :
		NO = (ite/Time_interval)
		VFile 		= HDF5File(mesh.mpi_comm(),"solution/velocity/u%s.h5" %NO,"w")
		DFile 		= HDF5File(mesh.mpi_comm(),"solution/displacement/d%s.h5"%NO,"w")
		PFile 		= HDF5File(mesh.mpi_comm(),"solution/pressure/p%s.h5"%NO,"w")
		V0File 		= HDF5File(mesh.mpi_comm(),"solution/velocity0/u0%s.h5"%NO,"w")
		WFile 		= HDF5File(mesh.mpi_comm(),"solution/meshvel/w%s.h5"%NO,"w")

		VFile.write(v,"vel" ,ite)
		DFile.write(U,"dis" ,ite)
		PFile.write(p,"pre" ,ite)
		V0File.write(v0,"vel0" ,ite)
		WFile.write(w,"mesh_vel" ,ite)
		VFile.close()
		PFile.close()
		DFile.close()
		V0File.close()
		WFile.close()
		if MPI.rank(MPI.comm_world) == 0 :
			print ('time = ', (t*d/u_av))
			print ('ite_no = ' , (ite))

	v0.assign(v)
	#Ue  = project(Ue0,DIGI,solver_type='cg',preconditioner_type='amg')
	t += float(k)
	ite += 1

print ("elapsed CPU time: ", (time.process_time() - timer0))  
