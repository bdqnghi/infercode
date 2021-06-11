import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_type_dim', type=int, default=30,
                        help='node type dimension size')
    parser.add_argument('--node_token_dim', type=int,
                        default=30, help='node token dimension size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
    parser.add_argument('--verbal', type=bool, default=True,
                        help='print training info or not')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--log_path', default="logs/",
                        help='log path for tensorboard')
    parser.add_argument('--checkpoint_every', type=int,
                        default=100, help='check point to save model')
    parser.add_argument('--validating', type=int,
                        default=1, help='validating or not')
    parser.add_argument('--tree_size_threshold_upper', type=int,
                        default=5000, help='tree size threshold')
    parser.add_argument('--tree_size_threshold_lower', type=int,
                        default=30, help='tree size threshold')                   
    parser.add_argument('--data_directory', default="java-small/training",
                        help='path of data directory to pre-process')
    parser.add_argument('--subtree_directory', default="java-small-subtrees/training",
                        help='path of data directory to pre-process')
    parser.add_argument('--label_path', default="OJ_stmt_train_test_val/train",
                        help='path of label to predict')
    parser.add_argument('--node_type_vocabulary_path', default="vocab/type_vocab.csv",
                        help='the path to node type vocab')
    parser.add_argument('--token_vocabulary_path', default="vocab/token_vocab.csv",
                        help='the path to node token vocab')
    parser.add_argument('--subtree_vocabulary_path', default="subtree_features/OJ_stmt_features_train.csv",
                        help='the path to subtree vocab')
    parser.add_argument('--dataset', default="java-small",
                        help='name of dataset')
    opt = parser.parse_args()

    return opt