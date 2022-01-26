import os

import numpy as np

from nn_analysis import utils
            
def main():
    print(f"Process PID: {os.getpid()}")
    tmp_filename_A = "tmp/tmp_file_A.pkl"
    tmp_filename_B = "tmp/tmp_file_B.pkl"
    print("Creating objects...")
    A = np.random.normal(size=(10000,5000))
    B = np.random.normal(size=(10000,5000))
    print("Finished creating object.")
    with utils.SimulFileHandler(tmp_filename_A, tmp_filename_B):
        print("Dumping A...")
        utils.save_data(tmp_filename_A, A)
        print("Done A. Dumping B...")
        utils.save_data(tmp_filename_B, B)
        print("Done B.")

if __name__ == '__main__':
    main()