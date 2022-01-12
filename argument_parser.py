"""Script to parse all the command-line arguments"""
import argparse


def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def argument_parser():
    """Function to parse all the arguments"""

    parser = argparse.ArgumentParser(description='Block Model')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='ADD')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='ADD')
    parser.add_argument('--num_inp_heads', type=int, default=1, metavar='E', help='num of heads in input attention')
    parser.add_argument('--sequence_length', type=int, default=51, metavar='S',
                        help='ADD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='ADD')
    parser.add_argument('--input_dropout', type=float, default=0.5, metavar='dropout', help='dropout')
    parser.add_argument('--comm_dropout', type=float, default=0.5)
    parser.add_argument('--kl_coeff', type=float, default=0.0, metavar='KL_coeff',
                        help='KL_coeff')
    parser.add_argument('--num_units', type=int, default=6, metavar='num_blocks',
                        help='Number_of_units')
    parser.add_argument('--num_encoders', type=int, default=1, metavar='num_encoders',
                        help='Number of encoders ')
    parser.add_argument('--k', type=int, default=4, metavar='topk',
                       help='Number_of_topk_blocks')
    parser.add_argument('--memorytopk', type=int, default=4, metavar='memtopk',
                       help='Number_of_topk_blocks')

    parser.add_argument('--hidden_size', type=int, default=600, metavar='hsize',
                        help='hidden_size')
    parser.add_argument('--n_templates', type=int, default=0, metavar='shared_blocks',
                        help='num_templates')
    #parser.add_argument('--num_modules_read_input', type=int, default=4, metavar='sort of proxy to inp heads')
    parser.add_argument('--share_inp', type=str2bool, default=False, metavar='share inp rims parameters')
    parser.add_argument('--share_comm', type=str2bool, default=False, metavar='share comm rims parameters')


    parser.add_argument('--do_rel', type=str2bool, default=False, metavar='use relational memory or not?')
    parser.add_argument('--memory_slots', type=int, default=4, metavar='memory slots for rel memory')
    parser.add_argument('--memory_mlp', type=int, default=4, metavar='no of memory mlp for rel memory')
    #parser.add_argument('--num_memory_heads', type=int, default=4, metavar='memory heads for rel memory')
    #parser.add_argument('--memory_head_size', type=int, default=16, metavar='memory head size for rel memory')

    parser.add_argument('--attention_out', type=int, default=340, help='ADD')

    parser.add_argument('--id', type=str, default='default',
                        metavar='id of the experiment', help='id of the experiment')
    parser.add_argument('--rnn_cell', type=str, default='blocks',
                        metavar='algorithm of the experiment', help='one of LSTM,GRU or blocks')
    parser.add_argument('--model_persist_frequency', type=int, default=20,
                        metavar='Frequency at which the model is persisted',
                        help='Number of training epochs after which model is to '
                             'be persisted. -1 means that the model is not'
                             'persisted')
    parser.add_argument('--batch_frequency_to_log_heatmaps', type=int, default=-1,
                        metavar='Frequency at which the heatmaps are persisted',
                        help='Number of training batches after which we will persit the '
                             'heatmaps. -1 means that the heatmap will not be'
                             'persisted')
    parser.add_argument('--path_to_load_model', type=str, default="",
                        metavar='Relative Path to load the model',
                        help='Relative Path to load the model. If this is empty, no model'
                             'is loaded.')
    parser.add_argument('--components_to_load', type=str, default="",
                        metavar='_ seperated list of model components that are '
                                'to be loaded.',
                        help='_ (underscore) seperated list of model components '
                             'that are to be loaded. Possible components '
                             'are blocks, encoders, decoders and rules - '
                             'eg blocks, blocks_rules, rules_blocks, rules,'
                             'rules_encoders or encoders etc', )

    parser.add_argument('--train_dataset', type=str, default="balls4mass64.h5",
                        metavar='path to dataset on which the model should be '
                                'trained',
                        help='path to dataset on which the model should be '
                             'trained')
    parser.add_argument('--test_dataset', type=str,
                        metavar='path to dataset on which the model should be '
                                'tested for stove',
                        help='path to dataset on which the model should be '
                             'tested for stove')

    parser.add_argument('--transfer_dataset', type=str, default="balls678mass64.h5",
                        metavar='path to dataset on which the model should be '
                                'transfered',
                        help='path to dataset on which the model should be '
                             'transfered')

    parser.add_argument('--should_save_csv', type=str2bool, nargs='?',
                        const=True, default=True,
                        metavar='Flag to decide if the csv logs should be created. '
                                'It is useful as creating csv logs makes a lot of'
                                'files.',
                        help='Flag to decide if the csv logs should be created. '
                             'It is useful as creating csv logs makes a lot of'
                             'files.')

    parser.add_argument('--should_resume', type=str2bool, nargs='?',
                        const=True, default=False,
                        metavar='Flag to decide if the previous experiment should be '
                                'resumd. If this flag is set, the last saved model '
                                '(corresponding to the given id is fetched)',
                        help='Flag to decide if the previous experiment should be '
                                'resumd. If this flag is set, the last saved model '
                                '(corresponding to the given id is fetched)',)

    parser.add_argument('--experiment_name', type=str, default='4Balls')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--do_comm', type=str2bool, default=True)
    # added by Nan
    parser.add_argument('--input_key_size', type=int)
    parser.add_argument('--input_value_size', type=int)
    parser.add_argument('--input_query_size', type=int)
    parser.add_argument('--comm_key_size', type=int)
    parser.add_argument('--comm_value_size', type=int)
    parser.add_argument('--comm_query_size', type=int)
    parser.add_argument('--num_comm_heads', type=int, default=4)
    



    args = parser.parse_args()

    args.frame_frequency_to_log_heatmaps = 5

    args.folder_log = f"./logs/{args.id}"

    #if args.num_encoders != 1:
    #    args.num_encoders = args.num_blocks

    return args
