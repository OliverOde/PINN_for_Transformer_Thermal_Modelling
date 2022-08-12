# Prepare data helper function

import numpy as np
from pyDOE import lhs
from scipy.interpolate import interp1d

def prepare_data(df, T_tog, N_u, N_f):

    t = np.array(df['t'][0:len(T_tog)]).flatten()[:,None]
    x = np.linspace(0, 1, num=52)
    Tamb = np.array(df['Tamb'][0:len(T_tog)]).flatten()[:,None]
    Ttop = np.array(df['Ttop'][0:len(T_tog)]).flatten()[:,None]
    K = np.array(df['K'][0:len(T_tog)]).flatten()[:,None]
    PK = (K**2)*83

    X, T = np.meshgrid(x,t)
    X1, PK1 = np.meshgrid(x,PK)
    X2, Tamb1 = np.meshgrid(x,Tamb)
    X3, Ttop1 = np.meshgrid(x,Ttop)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = T_tog.flatten()[:,None]
    PK_star = np.hstack((X1.flatten()[:,None], PK1.flatten()[:,None]))
    Tamb_star = np.hstack((X2.flatten()[:,None], Tamb1.flatten()[:,None]))
    Ttop_star = np.hstack((X3.flatten()[:,None], Ttop1.flatten()[:,None]))
    PK_tog = np.repeat(PK, repeats=52, axis=1)
    Tamb_tog = np.repeat(Tamb, repeats=52, axis=1)
    Ttop_tog = np.repeat(Ttop, repeats=52, axis=1)

    lb = X_star.min(0)
    ub = X_star.max(0)

    # Initial and boundary conditions
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = T_tog[:,0:1]
    PK2 = PK1[:,0:1]
    Tamb2 = Tamb1[:,0:1]
    Ttop2 = Ttop1[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = T_tog[:,-1:]
    PK3 = PK1[:,-1:]
    Tamb3 = Tamb1[:,-1:]
    Ttop3 = Ttop1[:,-1:]

        # Training variables
    X_u_train_all = np.vstack([xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)   # latin hypercube sampling
    u_train_all = np.vstack([uu2, uu3])
    PK_train_all = np.vstack([PK2, PK3])
    Tamb_train_all = np.vstack([Tamb2, Tamb3])
    Ttop_train_all = np.vstack([Ttop2, Ttop3])

        # Randomization
    idx = np.random.choice(X_u_train_all.shape[0], N_u, replace=False) # random sampling for boundary training points
    X_u_train = X_u_train_all[idx, :] # training points for space and time coordinates at boundaries
    u_train = u_train_all[idx,:] # training points for temperature at boundaries
    PK_train = PK_train_all[idx,:] # training points for load loss at boundaries
    Tamb_train = Tamb_train_all[idx,:] # training points for ambient temperature at boundaries
    Ttop_train = Ttop_train_all[idx,:] # training points for top-oil temperature at boundaries
    
    # Second order interpolation for collocation points 
    f2 = interp1d(t[:,0], PK_train_all[0:len(T_tog)][:,0], kind='quadratic')
    PK_f = f2(X_f_train[:, 1])
    PK_f = np.reshape(PK_f, (len(PK_f), 1))
    f2 = interp1d(t[:,0], Tamb_train_all[0:len(T_tog)][:,0], kind='quadratic')
    Tamb_f = f2(X_f_train[:, 1])
    Tamb_f = np.reshape(Tamb_f, (len(Tamb_f), 1))
    f2 = interp1d(t[:,0], Ttop_train_all[0:len(T_tog)][:,0], kind='quadratic')
    Ttop_f = f2(X_f_train[:, 1])
    Ttop_f = np.reshape(Ttop_f, (len(Ttop_f), 1))
    
    # Get PK and Tamb for X_star
    X_int = np.round(X_star)
    indx = []
    for row in X_int:
        indx.append(np.where(np.all(row==X_u_train_all ,axis=1))[0][0])
    PK_list = []
    Tamb_list = []
    Ttop_list = []
    for i in range(len(indx)):
        PK_list.append(PK_train_all[indx[i]][0])
        Tamb_list.append(Tamb_train_all[indx[i]][0])
        Ttop_list.append(Ttop_train_all[indx[i]][0])

    PK_star = np.reshape(np.array(PK_list), (len(PK_list), 1))
    Tamb_star = np.reshape(np.array(Tamb_list), (len(Tamb_list), 1))
    Ttop_star = np.reshape(np.array(Ttop_list), (len(Ttop_list), 1))
    
    return X_f_train, X_u_train, u_train, PK_train, Tamb_train, Ttop_train, lb, ub, X_star, u_star, \
            PK_star, Tamb_star, Ttop_star, PK_tog, Tamb_tog, X, T, x, t, PK_f, Tamb_f, Ttop_f
