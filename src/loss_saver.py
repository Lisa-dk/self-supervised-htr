import time, os
import numpy as np
import csv
from datetime import datetime

class LossSaver:
    def __init__(self, dir, mode, batch_size):
        self.current_time = datetime.now()
        self.current_time = self.current_time.strftime("%d-%m-%Y_%H:%M:%S")
        self.filename = 'log_' + mode + '_' + str(batch_size) + '_' + str(self.current_time) + ".csv"
        self.filename = self.filename.replace(':', '-')
        self.directory = './saved_losses/saved_losses/' + dir
        os.makedirs(self.directory, exist_ok=True)
        

    def save_to_csv(self, epoch, loss, cer, wer):
        with open((self.directory + '/' + self.filename), "a") as file:
            loss, cer, wer = np.array2string(np.array(loss)), np.array2string(np.array(cer)), np.array2string(np.array(wer))
            
            file.write(f"{epoch} {loss} {cer} {wer}\n")

    def close(self):
        self.file.close()

if __name__ == "__main__":
    loss_saver = LossSaver("", 'train', 8)
    fake = [8.9, 0.04, 0.05]
    fake_test = [0.8, 9.9, 9.8]
    loss_saver.save_to_csv(0, 1.0, 0.8, 0.8)
    loss_saver.save_to_csv(0, 1.0, 0.8, 0.8)

    # loss_saver.close()







        
