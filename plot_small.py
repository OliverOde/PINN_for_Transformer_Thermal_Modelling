# Function Helper for Plotting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_small(filename, x, t, T_tog, U_pred, time_list):

    def figsize(scale, nplots = 1):
        fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = nplots*fig_width*golden_mean              # height in inches
        fig_size = [fig_width*1.32, fig_height*2.48] #[fig_width*1.77, fig_height*3.21] 
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
    gs0.update(top=1-0.05, bottom=1-1/5 +0.03, left=0.14, right=0.85, wspace=0)
    ax = plt.subplot(gs0[0, 0])
    h = ax.imshow(T_tog.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlabel('t', fontsize = 10)
    ax.set_ylabel('x', fontsize = 10)
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u_{FVM}(x,t)$, FVM solution', fontsize = 10)

    # Row 2   
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=1 - 1/5 - 0.05, bottom=1-2/5 + 0.03, left=0.14, right=0.85, wspace=0)
    ax = plt.subplot(gs1[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    for i in range(len(time_list)):
        ax.plot(t[time_list[i]]*np.ones((2,1)), line, 'w-', linewidth = 1)  

    ax.set_xlabel('t', fontsize = 10)
    ax.set_ylabel('x', fontsize = 10)
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$\hat{u}(x,t)$, PINN solution', fontsize = 10)

    # Row 3    
    gs2 = gridspec.GridSpec(1, len(time_list))
    gs2.update(top=1-2/5 - 0.05, bottom=1-3/5 , left=0.1, right=0.9, wspace=0.5)

    for j in range(len(time_list)):
        ax = plt.subplot(gs2[0, j])
        ax.plot(x,T_tog[time_list[j],:], 'b-', linewidth = 2, label = 'FVM')       
        ax.plot(x,U_pred[time_list[j],:], 'r--', linewidth = 2, label = 'PINN')
        ax.set_xlabel('x', fontsize = 10)
        ax.set_ylabel('T(x,t)', fontsize = 10)    
        ax.set_title('t = ' + str(time_list[j]), fontsize = 11)
        if j == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        #ax.legend()
        #ax.set_xlim([-0.1,1.1])
        #ax.set_ylim([-7,35])
    savefig(filename + '_small_' + str(time_list)) 
    
