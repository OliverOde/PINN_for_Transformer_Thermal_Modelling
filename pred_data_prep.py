# Prepare data helper function

import numpy as np

def pred_data_prep(df, T_tog_preds, start_time):

    t_preds = np.array(df['t'][start_time:start_time+len(T_tog_preds)]).flatten()[:,None]
    x_preds = np.linspace(0, 1, num=52)

    Tamb_preds = np.array(df['Tamb'][start_time:start_time+len(T_tog_preds)]).flatten()[:,None]
    Ttop_preds = np.array(df['Ttop'][start_time:start_time+len(T_tog_preds)]).flatten()[:,None]
    K_preds = np.array(df['K'][start_time:start_time+len(T_tog_preds)]).flatten()[:,None]
    PK_preds = (K_preds**2)*83

    X_preds, T_preds = np.meshgrid(x_preds, t_preds)
    X1_preds, PK1_preds = np.meshgrid(x_preds, PK_preds)
    X2_preds, Tamb1_preds = np.meshgrid(x_preds, Tamb_preds)
    X3_preds, Ttop1_preds = np.meshgrid(x_preds, Ttop_preds)

    X_star_preds = np.hstack((X_preds.flatten()[:,None], T_preds.flatten()[:,None]))
    u_star_preds = T_tog_preds.flatten()[:,None]
    PK_star_preds = np.hstack((X1_preds.flatten()[:,None], PK1_preds.flatten()[:,None]))
    Tamb_star_preds = np.hstack((X2_preds.flatten()[:,None], Tamb1_preds.flatten()[:,None]))
    Ttop_star_preds = np.hstack((X3_preds.flatten()[:,None], Ttop1_preds.flatten()[:,None]))
    PK_tog_preds = np.repeat(PK_preds, repeats=52, axis=1)
    Tamb_tog_preds = np.repeat(Tamb_preds, repeats=52, axis=1)
    Ttop_tog_preds = np.repeat(Ttop_preds, repeats=52, axis=1)

    # Initial and boundary conditions
    xx2_preds = np.hstack((X_preds[:,0:1], T_preds[:,0:1]))
    uu2_preds = T_tog_preds[:,0:1]
    PK2_preds = PK1_preds[:,0:1]
    Tamb2_preds = Tamb1_preds[:,0:1]
    Ttop2_preds = Ttop1_preds[:,0:1]
    xx3_preds = np.hstack((X_preds[:,-1:], T_preds[:,-1:]))
    uu3_preds = T_tog_preds[:,-1:]
    PK3_preds = PK1_preds[:,-1:]
    Tamb3_preds = Tamb1_preds[:,-1:]
    Ttop3_preds = Ttop1_preds[:,-1:]

    # Training variables
    X_u_train_all_preds = np.vstack([xx2_preds, xx3_preds])
    u_train_all_preds = np.vstack([uu2_preds, uu3_preds])
    PK_train_all_preds = np.vstack([PK2_preds, PK3_preds])
    Tamb_train_all_preds = np.vstack([Tamb2_preds, Tamb3_preds])
    Ttop_train_all_preds = np.vstack([Ttop2_preds, Ttop3_preds])
    
    # Get PK, Tamb and Ttop for X_star
    X_int = np.round(X_star_preds)
    indx = []
    for row in X_int:
        indx.append(np.where(np.all(row==X_u_train_all_preds ,axis=1))[0][0])
    PK_list = []
    Tamb_list = []
    Ttop_list = []
    for i in range(len(indx)):
        PK_list.append(PK_train_all_preds[indx[i]][0])
        Tamb_list.append(Tamb_train_all_preds[indx[i]][0])
        Ttop_list.append(Ttop_train_all_preds[indx[i]][0])

    PK_star_preds = np.reshape(np.array(PK_list), (len(PK_list), 1))
    Tamb_star_preds = np.reshape(np.array(Tamb_list), (len(Tamb_list), 1))
    Ttop_star_preds = np.reshape(np.array(Ttop_list), (len(Ttop_list), 1))

    return X_star_preds, PK_star_preds, Tamb_star_preds, Ttop_star_preds, X_preds, T_preds, u_star_preds
