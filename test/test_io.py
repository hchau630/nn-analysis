import os
import json

import numpy as np

from nn_analysis import utils
            
def main():
    print(f"Process PID: {os.getpid()}")
    tmp_path = "tmp"
    print("Creating object...")
    A = {
        'a': {
            'b': {
                'c': 'd'
            },
            'e': 'f',
        },
        'g': 'h',
    }
    print("Finished creating object.")
    print("Starting to dump...")
    utils.save_data(tmp_path, A, depth=0)
    print("Finished dumping")
    data = utils.load_data(tmp_path)
    print(json.dumps(data, indent=4))

if __name__ == '__main__':
    main()