import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int,
                    help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int,
                        default=4, help='input batch size')
    parser.add_argument('--train_batch_size', type=int,
                        default=5, help='train input batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='test input batch size')
    parser.add_argument('--val_batch_size', type=int,
                        default=5, help='val input batch size')
    parser.add_argument('--node_type_dim', type=int, default=30,
                        help='node type dimension size')
    parser.add_argument('--node_token_dim', type=int,
                        default=30, help='node token dimension size')
    parser.add_argument('--hidden_layer_size', type=int,
                        default=100, help='size of hidden layer')
    parser.add_argument('--num_hidden_layer', type=int,
                        default=1, help='number of hidden layer')
    parser.add_argument('--n_steps', type=int, default=10,
                        help='propagation steps number of GGNN')
    parser.add_argument('--n_edge_types', type=int, default=7,
                        help='number of edge types')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
    parser.add_argument('--verbal', type=bool, default=True,
                        help='print training info or not')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model_path', default="model",
                        help='path to save the model')
    # parser.add_argument('--model_accuracy_path', default="model_accuracy/method_name.txt",
    #                     help='path to save the the best accuracy of the model')
    parser.add_argument('--n_hidden', type=int, default=50,
                        help='number of hidden layers')
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
    parser.add_argument('--sampling_size', type=int,
                        default=60, help='sampling size for each epoch')
    parser.add_argument('--best_f1', type=float,
                        default=0.0, help='best f1 to save model')
    parser.add_argument('--aggregation', type=int, default=1, choices=range(0, 4),
                        help='0 for max pooling, 1 for attention with sum pooling, 2 for attention with max pooling, 3 for attention with average pooling')
    parser.add_argument('--distributed_function', type=int, default=0,
                        choices=range(0, 2), help='0 for softmax, 1 for sigmoid')
    parser.add_argument('--data_path', default="OJ_pkl_train_test_val/train",
                        help='path of data pickle')
    parser.add_argument('--label_path', default="OJ_stmt_train_test_val/train",
                        help='path of label to predict')
    parser.add_argument('--node_type_vocabulary_path', default="vocab/type_vocab.csv",
                        help='the path to node type vocab')
    parser.add_argument('--token_vocabulary_path', default="vocab/token_vocab.csv",
                        help='the path to node token vocab')
    parser.add_argument('--subtree_vocabulary_path', default="subtree_features/OJ_stmt_features_train.csv",
                        help='the path to subtree vocab')
    parser.add_argument('--task', type=int, default=1,
                        choices=range(0, 2), help='0 for training, 1 for testing')
    parser.add_argument('--loss', type=int, default=0,
                        choices=range(0, 3), help='loss function, 0 for softmax, 1 for sampled softmax, 2 for nce')
    parser.add_argument('--num_sampling', type=int, default=2, help="number of subtrees to be sampled for sampled softmax")
    parser.add_argument('--num_files_threshold', type=int, default=20000)
    parser.add_argument('--num_conv', type=int, default=1)
    parser.add_argument('--output_size', type=int, default=30)
    parser.add_argument('--dataset', default="OJ_new")
    parser.add_argument('--include_token', type=int,
                        default=1, help='including token for initializing or not, 1 for including, 0 for excluding')
    opt = parser.parse_args()

    return opt