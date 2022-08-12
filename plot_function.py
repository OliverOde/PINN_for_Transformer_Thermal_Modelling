# Function Helper for Plotting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_function(filename, x, t, T_tog, U_pred, X_u_train, u_train, mse, error_u, error_u_top, mse_u, mse_f, time_list, log_freq, mse_PK = None, mse_Tamb = None):

    def figsize(scale, nplots = 1):
        fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = nplots*fig_width*golden_mean              # height in inches
        fig_size = [fig_width*2, fig_height*4]
        return fig_size

    # I make my own newfig and savefig functions
    def newfig(width, nplots = 1):
        fig = plt.figure(figsize=figsize(width, nplots))
        ax = fig.add_subplot(111)
        return fig, ax

    # function to save figures
    def savefig(filename, crop = True):
        if crop == True:
            plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('{}.pdf'.format(filename))

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    # Row 1
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=1-0.05, bottom=1-1/5 + 0.05, left=0.14, right=0.85, wspace=0)
    ax = plt.subplot(gs0[0, 0])
    h = ax.imshow(T_tog.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u_{FVM}(x,t)$, FVM solution', fontsize = 12)

    # Row 2   
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=1 - 1/5 - 0.05, bottom=1-2/5 + 0.05, left=0.14, right=0.85, wspace=0)
    ax = plt.subplot(gs1[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Boundary points (%d)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    for i in range(len(time_list)):
        ax.plot(t[time_list[i]]*np.ones((2,1)), line, 'w-', linewidth = 1)  

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$\hat{u}(x,t)$, PINN solution', fontsize = 12)

    # Row 3    
    gs2 = gridspec.GridSpec(1, len(time_list))
    gs2.update(top=1-2/5 - 0.03, bottom=1-3/5 + 0.03, left=0.05, right=0.95, wspace=0.5)

    for j in range(len(time_list)):
        ax = plt.subplot(gs2[0, j])
        ax.plot(x,T_tog[time_list[j],:], 'b-', linewidth = 2, label = 'FVM')       
        ax.plot(x,U_pred[time_list[j],:], 'r--', linewidth = 2, label = 'PINN')
        ax.set_xlabel('x')
        ax.set_ylabel('T(x,t)')    
        ax.set_title('t = ' + str(time_list[j]), fontsize = 11)
        ax.legend()
        #ax.set_xlim([-0.1,1.1])
        #ax.set_ylim([-7,35])
 
    # Row 4
    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(top=1-3/5 - 0.03, bottom=1-4/5 + 0.03, left=0.3, right=0.7, wspace=0.5)
    ax = plt.subplot(gs3[0, 0])
    ax.plot(mse)
    plt.ylabel("Total mse")
    plt.xlabel("Epochs/Iterations")
    plt.yscale('log')

    # Row 5
    gs4 = gridspec.GridSpec(1, 2)
    gs4.update(top=1-4/5 - 0.03, bottom=0 + 0.03, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs4[0, 0])
    ax.plot(error_u)
    plt.ylabel("Error u")
    plt.xlabel("Epochs (per " + str(log_freq) + ")")
    plt.yscale('log')

    ax = plt.subplot(gs4[0, 1])
    ax.plot(error_u_top)
    plt.ylabel("Error u top")
    plt.xlabel("Epochs (per " + str(log_freq) + ")")
    plt.yscale('log')

    savefig(filename + str(time_list)) 

    # Second PDF of MSE 
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    # ROW 1
    gs0_2 = gridspec.GridSpec(1, 2)
    gs0_2.update(top=1 - 0.03, bottom= 1-1/5 + 0.03, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs0_2[0, 0])
    ax.plot(mse_u)
    plt.ylabel("MSE u")
    plt.xlabel("Epochs/Iterations")
    plt.yscale('log')

    ax = plt.subplot(gs0_2[0, 1])
    ax.plot(mse_f)
    plt.ylabel("MSE Residual")
    plt.xlabel("Epochs/Iterations")
    plt.yscale('log')
    
    if mse_PK is not None:
        # ROW 2
        gs1_2 = gridspec.GridSpec(1, 2)
        gs1_2.update(top=1-1/5 - 0.03, bottom= 3/5 + 0.03, left=0.1, right=0.9, wspace=0.5)

        ax = plt.subplot(gs1_2[0, 0])
        ax.plot(mse_PK)
        plt.ylabel("MSE PK")
        plt.xlabel("Epochs/Iterations")
        plt.yscale('log')

        ax = plt.subplot(gs1_2[0, 1])
        ax.plot(mse_Tamb)
        plt.ylabel("MSE Tamb")
        plt.xlabel("Epochs/Iterations")
        plt.yscale('log')

    savefig(filename + "_MSE")    
