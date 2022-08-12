# Function Helper for Plotting Collocation Points

import numpy as np
import matplotlib.pyplot as plt

def plot_collocation(filename, X_f_train, col_weights):
    
    # function to save figures
    def savefig(filename, crop = False):
        if crop == True:
            plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('{}.pdf'.format(filename))
            
    plt.scatter(X_f_train[:, 1], X_f_train[:, 0], c = col_weights.numpy(), s = col_weights.numpy()/5)
    plt.xlabel('t')
    plt.ylabel('x')
    savefig(filename) 
