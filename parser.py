import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('Neural Predicting Higher-order Patterns in Temporal Networks')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use',
                        choices=['DAWN', 'tags-ask-ubuntu', 'tags-math-sx', 'NDC-substances', 'congress-bills', 'threads-ask-ubuntu'],
                        default='tags-math-sx')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')
    parser.add_argument('-nega', '--negative', type=str, default='Nega20', choices=['Nega20', 'NegaWedge', 'rand'], help='negative sampling(last 20%) and randomly choosing(rand)')
    parser.add_argument('--data_usage', default=1.0, type=float, help='portion of data to use (0-1)')

    # methodology-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=['64', '2'],
                        help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=1, help='number of network layers')
    parser.add_argument('--bias', default=1e-5, type=float, help='alpha for TRWs, controlling sampling preference with time closeness, default to 0 which is uniform sampling')
    parser.add_argument('--agg', type=str, default='walk', choices=['tree', 'walk'],
                        help='tree based hierarchical aggregation or walk-based flat lstm aggregation')
    parser.add_argument('--pos_enc', type=str, default='lp', choices=['spd', 'lp', 'saw','concat', 'sum_pooling', 'sum_pooling_after'], help='way to encode distances, shortest-path distance or lp means counting, self-based anonymous walk (baseline), concat, sum_pooling and sum_pooling_after are baselines')
    parser.add_argument('--pos_dim', type=int, default=108, help='dimension of the positional embedding(model dim mod 16 == 0); 12 <=> no distance encoding, used for ablation study')
    parser.add_argument('--pos_sample', type=str, default='multinomial', choices=['multinomial', 'binary'], help='two practically different sampling methods that are equivalent in theory ')
    parser.add_argument('--walk_pool', type=str, default='attn', choices=['attn', 'sum'], help='how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other walk_ arguments')
    parser.add_argument('--walk_n_head', type=int, default=8, help="number of heads to use for walk attention")
    parser.add_argument('--walk_mutual', action='store_true', default=False, help="whether to do mutual query for source and target node random walks")
    parser.add_argument('--walk_linear_out', action='store_true', default=False, help="whether to linearly project each node's ")

    parser.add_argument('--attn_agg_method', type=str, default='attn', choices=['attn', 'lstm', 'mean'], help='local aggregation method, we only use the default here')
    parser.add_argument('--attn_mode', type=str, default='prod', choices=['prod', 'map'],
                        help='use dot product attention or mapping based, we only use the default here')
    parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
    parser.add_argument('--time', type=str, default='time', choices=['time', 'pos', 'empty'], help='how to use time information, we only use the default here')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='toleratd margainal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=2, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=6, help='which gpu to use')
    parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')

    parser.add_argument('--ablation', action='store_true', default=False, help='Interpretation or not')
    parser.add_argument('--ablation_type', type=int, default=0, help='Interpretation type: For interpretation, we have 4 tasks. 1: triangle vs wedge;')
    parser.add_argument('--interpretation', action='store_true', default=False, help='Interpretation or not')
    parser.add_argument('--interpretation_type', type=int, default=0, help='Interpretation type: For interpretation, we have 4 tasks. 1: closure vs trianlge; 2: triangle + closure vs wedge; 3: wedge and edge; 4: closure and wedge; Default 0 means no interpretation')
    parser.add_argument('--test_path', type=str, default=None, help='Best model File Path')
    parser.add_argument('--time_prediction', action='store_true', default=False, help='Time prediction task')
    parser.add_argument('--time_prediction_type', type=int, default=0, help='Interpretation type: For time_prediction, we have 3 tasks. 1 for closure; 2 for triangle; 3 for wedge;  Default 0 means no time_prediction')
    parser.add_argument('--debug', action='store_true', default=False, help='Time prediction task')

    # parameters for baselines
    parser.add_argument('--test_baselines', action='store_true', default=False, help='test baselines or not')
    # parser.add_argument('--data', required=True, help='Network name')
    parser.add_argument('--model', default='tgat', choices=['jodie', 'nhp', 'tgn', 'tgat', 'dyrep','JC', 'PA', 'AA', 'Arith', 'Geom', 'Harm', 'AA_Arith', 'AA_Benson', 'JC_Arith', 'JC_Benson', 'PA_Arith', 'PA_Benson', 'AA_Geom', 'PA_Geom', 'JC_Geom', 'four_clique', 'four_diamond'], help="Model name")
    


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv
