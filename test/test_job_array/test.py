import argparse

def main(model_name, layers, acts_name, version):
    print(model_name)
    print(layers)
    print(acts_name)
    print(version)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save activations')
    parser.add_argument('model_name', type=str,
                        help='Specify the model of which activations are saved')
    parser.add_argument('layers', type=int, nargs='+',
                        help='Specify the layers of which activations are saved')
    parser.add_argument('acts_name', type=str,
                        help='Specify what activations are saved')
    parser.add_argument('version', type=int,
                        help='Version number. Error will be raised if the activations for the specified version' \
                             'already exists and --overwrite flag is not provided.')
    args = parser.parse_args()
    main(args.model_name, args.layers, args.acts_name, args.version)