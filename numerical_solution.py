# Numerical Scheme
import numpy as np

def numerical_solution(param_df, Nx, c=2000):

  # Parameters
  k = 50
  rho = 900
  #c = 2000
  H = 1
  h = 1000
  P0 = 15000

  timesteps = len(param_df)

  K_np = param_df['K'].values
  Tamb_np = param_df['Tamb'].values
  Ttop_np = param_df['Ttop'].values

  PK_np = 83000*(K_np**2)
  Tamb_np = np.reshape(np.array(Tamb_np), (len(Tamb_np), 1))
  Ttop_np = np.reshape(np.array(Ttop_np), (len(Ttop_np), 1))
  PK_np = np.reshape(np.array(PK_np), (len(PK_np), 1))

  # Time and space data and discretization 
  time_sec = timesteps*3600 # Seconds
  dt = time_sec/timesteps # time-step
  dx = H/Nx;  # space-step

  x = np.linspace(0.01, 1.01, 50, endpoint=False)
  x = np.append(x, 1)
  x = np.append(0, x)

  # Create A matrix
  A = np.zeros((Nx, Nx))
  A = np.diag(rho*c*dx/dt + 2*k/dx + h*dx*np.ones((Nx))) + np.diag(-(k/dx)*np.ones(Nx-1), 1) + np.diag(-(k/dx)*np.ones(Nx-1), -1)
  A[0, 0] = rho*c*dx/dt + 3*k/dx + h*dx
  A[Nx-1, Nx-1] = A[0, 0]

  T0 = np.linspace(Tamb_np[0], Ttop_np[0], Nx)
  B = np.zeros(Nx)

  T = np.zeros((Nx, timesteps))
  for i in range(timesteps):
    if i == 0:
      for j in range(Nx):
        B[j] = (rho*c*T0[j]*dx/dt + (P0+PK_np[0]+h*Tamb_np[0])*dx)
      B[0] = rho*c*T0[0]*dx/dt + (2*k/dx+h*dx)*Tamb_np[0] + (P0+PK_np[0])*dx
      B[-1] = rho*c*T0[Nx-1]*dx/dt + 2*k/dx*Ttop_np[0] + (P0+PK_np[0]+h*Tamb_np[0])*dx
    else:
      for j in range(Nx):
        B[j] = (rho*c*T[j, 0]*dx/dt + (P0+PK_np[i]+h*Tamb_np[i])*dx)
      B[0] = rho*c*T[0, 0]*dx/dt + (2*k/dx+h*dx)*Tamb_np[i] + (P0+PK_np[i])*dx
      B[-1] = rho*c*T[-1, -1]*dx/dt + 2*k/dx*Ttop_np[i] + (P0+PK_np[i]+h*Tamb_np[i])*dx 

    T[:, i] = np.linalg.solve(A, B)

  T_solution = np.append(np.transpose(Tamb_np), T, axis=0)
  T_solution = np.transpose(np.append(T_solution, np.transpose(Ttop_np), axis=0))
   
  return T_solution