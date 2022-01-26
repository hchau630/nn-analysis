import time
import os

import numpy as np

from nn_analysis import utils
            
def main():
    print(f"Process PID: {os.getpid()}")
    tmp_filename = "tmp/tmp_file.pkl"
    with utils.TmpFileHandler(tmp_filename):
        print("Creating object...")
        A = np.random.normal(size=(10000,5000))
        print("Finished creating object.")
        print("Starting to dump...")
        utils.save_data(tmp_filename, A)
        print("Finished dumping")
        time.sleep(20)

if __name__ == '__main__':
    main()