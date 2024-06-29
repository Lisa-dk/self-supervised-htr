import time, os
import numpy as np
import csv
from datetime import datetime

class LossSaver:
    def __init__(self, mode, batch_size, n_style):
        self.current_time = datetime.now()
        self.current_time = self.current_time.strftime("%d-%m-%Y_%H:%M:%S")
        self.filename = 'log_' + mode + '_' + str(batch_size) + '_' + str(n_style) + '_' + str(self.current_time) + ".csv"
        self.filename = self.filename.replace(':', '-')
        self.directory = './saved_losses/saved_losses'
        os.makedirs(self.directory, exist_ok=True)
        
    def idx_to_loss_type_train(idx):
        if idx == 0:
            return 'dis_tr'
        elif idx == 1:
            return 'dis'
        elif idx == 2:
            return 'cla_tr'
        elif idx == 3:
            return 'cla'
        elif idx == 4:
            return 'rec_tr'
        elif idx == 5:
            return 'rec'
        elif idx == 6:
            return 'l1'
        
    def idx_to_loss_type_eval(idx):
        if idx == 0:
            return 'dis'
        elif idx == 1:
            return 'cla'
        elif idx == 2:
            return 'rec'

    def save_to_csv(self, losses, mode):

        if mode == 'train':
            loss_types = ['dis_tr', 'dis', 'cla_tr', 'cla', 'rec_tr', 'rec', 'l1']
            idx = 0
            with open((self.directory + '/' + self.filename), "a") as file:
                for loss in losses[1:]:
                    arr = np.array(loss)
                    str_arr = np.array2string(arr, separator=',')
                    file.write(loss_types[idx] + ',' + str(losses[0]) + ',' + str_arr[1:-1] + '\n')
                    idx += 1
                
        elif mode == 'eval':

            loss_types = ['dis', 'cla', 'rec']
            idx = 0
            with open((self.directory + '/' + self.filename), "a") as file:
                for loss in losses[1:]:
                    arr = np.array(loss)
                    str_arr = np.array2string(arr, separator=',')
                    file.write(loss_types[idx] + ',' + str(losses[0]) + ',' + str_arr[1:-1] + '\n')
                    idx += 1



    def close(self):
        self.file.close()


if __name__ == "__main__":
    loss_saver = LossSaver('train', 12, 3)
    fake = [9, [33, 4],  [33, 4], [33, 4],  [33, 4], [33, 4],  [33, 4], [33, 5]]
    fake_test = [9, [33, 4],  [33, 4], [33, 4]]
    loss_saver.save_to_csv(fake_test, "eval")
    loss_saver.save_to_csv(fake, "train")

    # loss_saver.close()







        
