import sys
import os
import subprocess

from nn_analysis import utils
from nn_analysis.constants import ENV_CONFIG_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)

save_config_path = f"{env_config['script_tmp_path']}/save_config.txt"
analyze_config_path = f"{env_config['script_tmp_path']}/analyze_config.txt"

BIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(*args):
    try:
        print()
        
        # setup.py
        subprocess_args = ["python", f'{BIN_DIR}/py_scripts/main_setup.py', *args]
        print("Calling setup.py with command: ")
        print(" ".join(subprocess_args))
        print()

        out = subprocess.run(subprocess_args, capture_output=True, check=True)
        print(out.stdout.decode("utf-8"))
        print()

        # save_multi.sh
        if os.path.getsize(save_config_path) > 0:
            subprocess_args = [f'{BIN_DIR}/save_multi.sh', save_config_path]
            print("Calling save_multi.sh with command: ")
            print(" ".join(subprocess_args))
            print()

            out = subprocess.run(subprocess_args, capture_output=True, check=True)

            PID = out.stdout.decode("utf-8").split('\n')[-2] # get last line of output as PID (index -1 is '\n')
            print(f"save_multi.sh job PID {PID}")
        else:
            PID = None
            print("Save config file is empty. Skipping saving.")
        print()

        # analyze_multi.sh
        if os.path.getsize(analyze_config_path) > 0:
            if PID is None:
                subprocess_args = [f'{BIN_DIR}/analyze_multi.sh', analyze_config_path]
            else:
                subprocess_args = [f'{BIN_DIR}/analyze_multi_seq.sh', analyze_config_path, PID]
            print("Calling analyze_multi.sh with command: ")
            print(" ".join(subprocess_args))
            print()

            out = subprocess.run(subprocess_args, capture_output=True, check=True)

            PID = out.stdout.decode("utf-8").split('\n')[-2]  # get last line of output as PID (index -1 is '\n')
            print(f"analyze_multi.sh job PID {PID}")
        else:
            print("Analyze config file is empty. Skipping analyzing.")
        print()
            
    except subprocess.CalledProcessError as err:
        print("The subprocess called by the command")
        print(f"{err.cmd}")
        print(f"exited with non-zero exit code {err.returncode}")
        print("Output of that subprocess:")
        print(f"{err.stdout.decode('utf-8')}")
        print("Stderr output of that subprocess:")
        print(f"{err.stderr.decode('utf-8')}")

if __name__ == '__main__':
    print("Started main_save_and_analyze.py...")
    main(*sys.argv[1:])