# Logger for visualizing training

import json
import tensorflow as tf
import time
from datetime import datetime

class Logger(object):
    
    def __init__(self, hp):
        print("Hyperparameters:")
        print(json.dumps(hp, indent=2))
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.mse_u = []
        self.mse_PK = []
        self.mse_Tamb = []
        self.mse_f = []
        self.mse = []
        self.error_u_progress = []
        self.error_u_top_progress = []

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")
        self.model = model
        print(model.summary())

    def log_train_opt(self, name):
        print(f"======== Starting {name} optimization ========")

    def log_train_end(self, epoch):
        print("==================")
        errors = self.get_error_u()
        print(f"Training finished (epoch {epoch}): " +
              f"Duration = {self.get_elapsed()}  " +
              f"Error u = {errors[0]:.4e}  "  +
              f"Error u_top = {errors[1]:.4e}  " )
        
    def log_train_progress(self, epoch):
        errors = self.get_error_u()
        print(f"Epoch number: {epoch} " +
              f" Duration = {self.get_elapsed()}  " +
              f" Error u = {errors[0]:.4e}  "   +
              f" Error u_top = {errors[1]:.4e}  ")
        
        self.error_u_progress.append(errors[0])
        self.error_u_top_progress.append(errors[1])
        
    def append_losses(self, mse_u, mse_f, mse_PK = None, mse_Tamb = None):
        self.mse_u.append(mse_u)
        self.mse_f.append(mse_f)
        if mse_PK is not None:
            self.mse_PK.append(mse_PK)
            self.mse_Tamb.append(mse_Tamb)
            self.mse.append(mse_u + mse_PK + mse_Tamb + mse_f)
        else:
            self.mse.append(mse_u + mse_f)
        
    def get_losses_lists_4(self):
        return self.mse_u, self.mse_PK, self.mse_Tamb, self.mse_f, self.mse, self.error_u_progress, self.error_u_top_progress
    
    def get_final_losses_lists_4(self):
        return self.mse_u[-1], self.mse_PK[-1], self.mse_Tamb[-1], self.mse_f[-1], self.mse[-1]
    
    def get_losses_lists_2(self):
        return self.mse_u, self.mse_f, self.mse, self.error_u_progress, self.error_u_top_progress
    
    def get_final_losses_lists_2(self):
        return self.mse_u[-1], self.mse_f[-1], self.mse[-1]

    def plot_error_u(self):
        plt.plot(self.error_u_progress)
        plt.ylabel("Error u")
        plt.yscale('log')
        plt.xlabel("Epochs/Iterations (per " + str(self.frequency) + " epochs)")
        plt.show()

    def plot_error_u_top(self):
        plt.plot(self.error_u_top_progress)
        plt.ylabel("Error u top")
        plt.yscale('log')
        plt.xlabel("Epochs/Iterations (per " + str(self.frequency) + " epochs)")
        plt.show()
    
    def plot_loss(self, loss, loss_string, log_yscale = True, log_xscale = False):
        plt.plot(loss)
        plt.ylabel(loss_string)
        plt.xlabel("Epochs/Iterations")
        if log_yscale:
            plt.yscale('log')
        if log_xscale:
            plt.xscale('log')
        plt.show()