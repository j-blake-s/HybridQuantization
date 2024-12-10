import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=11, help="number of classes")
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--interval', type=int, default=8)
    parser.add_argument('--batch_rate', type=int, default=1)
    ##TODO ask Blake about this core, I changed to 0
    parser.add_argument('--core', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--no_quant', action='store_true')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_name', type=str, default="default_model")
    parser.add_argument('--save_folder', type=str, default="./saved_models")
    parser.add_argument('--model', type=str, default="acnn")
    parser.add_argument('--lr', type=float, default=0.0001)



    args, _ = parser.parse_known_args()
    return args