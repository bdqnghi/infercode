import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int,
                        help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int,
                        default=4, help='input batch size')
    parser.add_argument('--label_size', type=int, default=104,
                    help='number of labels')
    parser.add_argument('--node_type_dim', type=int, default=30,
                        help='node type dimension size')
    parser.add_argument('--node_token_dim', type=int,
                        default=30, help='node token dimension size')
    parser.add_argument('--conv_output_dim', type=int,
                        default=30, help='size of convolutional output')
    parser.add_argument('--hidden_layer_size', type=int,
                        default=100, help='size of hidden layer')
    parser.add_argument('--num_hidden_layer', type=int,
                        default=1, help='number of hidden layer')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
    parser.add_argument('--verbal', type=bool, default=True,
                        help='print training info or not')
    parser.add_argument('--model_path', default="model",
                        help='path to save the model')
    parser.add_argument('--parser', default="pycparser",
                        help='name of the parser used to parse the code')
    parser.add_argument('--n_hidden', type=int, default=50,
                        help='number of hidden layers')
    parser.add_argument('--log_path', default="logs/",
                        help='log path for tensorboard')
    parser.add_argument('--checkpoint_every', type=int,
                        default=50, help='check point to save model')
    parser.add_argument('--validating', type=int,
                        default=1, help='validating or not')
    parser.add_argument('--tree_size_threshold_upper', type=int,
                        default=2800 , help='tree size threshold')
    parser.add_argument('--tree_size_threshold_lower', type=int,
                        default=30, help='tree size threshold')                   
    parser.add_argument('--sampling_size', type=int,
                        default=60, help='sampling size for each epoch')
    parser.add_argument('--best_f1', type=float,
                        default=0.0, help='best f1 to save model')
    parser.add_argument('--train_tree_path', default="OJ_rs/OJ_rs-buckets-train.pkl",
                        help='path of training data')
    parser.add_argument('--train_bucket_path', default="OJ_rs/OJ_rs-buckets-train.pkl",
                        help='path of training data')
    parser.add_argument('--node_type_vocabulary_path', default="vocab/Rust/node_type/txl_used.txt",
                        help='the path to node type vocab')
    parser.add_argument('--token_vocabulary_path', default="vocab/Rust/node_token/token.txt",
                        help='the path to node token vocab')
    parser.add_argument('--subtree_vocabulary_path', default="subtrees/subtrees.txt",
                        help='the path to subtre vocab')
    parser.add_argument('--task', type=int, default=1,
                        choices=range(0, 2), help='0 for training, 1 for testing')
    parser.add_argument('--num_files_threshold', type=int, default=20000)
    parser.add_argument('--num_conv', type=int, default=1)
    parser.add_argument('--model_name', default="stmt")
    parser.add_argument('--node_init', type=int,
                        default=1, help='including token for initializing or not, 1 for including only token, 0 for including only type, 2 for both')
    opt = parser.parse_args()

    return opt