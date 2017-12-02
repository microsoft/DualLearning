import argparse
parser = argparse.ArgumentParser()

# data I/O

parser.add_argument('--data_dir', type=str, default='./data/imdb.pkl', help='Location for the dataset')
parser.add_argument('--LMScoreFile', type=str, default='./data/LMScore.npz', help='Location for the LMScoreFile')
parser.add_argument('--GCRmode', dest='GCRmode', action='store_true', help='GCRmode')
parser.add_argument('--gpu', type=int, default=0, help='')


# optimization parameters
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--model_S2L', type=str, default='warmClassifier.npz')
parser.add_argument('--model_S2L_pkl', type=str, default=None)
parser.add_argument('--model_L2S', type=str, default='warmCLM.npz')
parser.add_argument('--model_L2S_pkl', type=str, default=None)
parser.add_argument('--dual_style', type=str, default='all', help='all | S2L | L2S ')
parser.add_argument('--optim', type=str, default='adadelta')

parser.add_argument('--minibatch', type=int, default=16, help='')
parser.add_argument('--trade_off_S2L', type=float, default=5e-3, help='the consistence tradeoff')
parser.add_argument('--trade_off_L2S', type=float, default=5e-3, help='the consistence tradeoff')
parser.add_argument('--clip_S2L', type=float, default=-1., help='gradient clip S2L')
parser.add_argument('--clip_L2S', type=float, default=5., help='gradient clip L2S')
parser.add_argument('--bias', type=float, default=0.02, help='the bias')
parser.add_argument('--FreezeEmb', dest='FreezeEmb', action='store_true', help='FreezeEmb')
parser.add_argument('--lrS2L', type=float, default=0.1, help='')
parser.add_argument('--lrL2S', type=float, default=0.1, help='the bias')
parser.add_argument('--lrate', type=float, default=0.1, help='the bias')
parser.add_argument('--maxEpoch', type=int, default=100, help='')
parser.add_argument('--validFreq', type=int, default=2000, help='')
parser.add_argument('--classifier_drop_in', type=float, default=0.8, help='classifier_drop_in')
parser.add_argument('--classifier_drop_out', type=float, default=0.5, help='classifier_drop_out')
parser.add_argument('--CLM_drop_in', type=float, default=0.5, help='CLM_drop_in')
parser.add_argument('--CLM_drop_out', type=float, default=0.5, help='CLM_drop_out')

config_params = parser.parse_args()







