import torch

def main():
    path = '/mnt/smb/locker/issa-locker/users/hc3190/data/models/checkpoints/moco/moco_v1_factorize2_v3/0049.pth.tar'
    # path = 'data/0049.pth.tar'

    with open(path, 'rb') as f:
        print("Success")

if __name__ == '__main__':
    main()