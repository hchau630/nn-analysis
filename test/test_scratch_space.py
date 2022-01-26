import time
import os

import numpy as np

from nn_analysis import utils

def main():
    print(f"Process PID: {os.getpid()}")
    print("Creating object...")
    A = np.random.normal(size=(10000,5000))
    print("Finished creating object.")
        
    print("Testing normal save location")
    tmp_filename = "tmp/tmp_file.pkl"
    with utils.TmpFileHandler(tmp_filename):
        print("Starting to dump...")
        start = time.time()
        utils.save_data(tmp_filename, A)
        print(f"Finished dumping. Time taken: {time.time()-start}")
        
    print("Testing scratch space save location")
    tmp_filename = "/local/hc3190/tmp_file.pkl" # need to create the folder hc3190 first
    with utils.TmpFileHandler(tmp_filename):
        print("Starting to dump...")
        start = time.time()
        utils.save_data(tmp_filename, A)
        print(f"Finished dumping. Time taken: {time.time()-start}")

if __name__ == '__main__':
    main()