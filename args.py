import argparse

import datetime
import os 
from utils import configs

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_dir',type=str,default=None)
    parser.add_argument('--train_dir',type=str,default=configs['TRAIN_DIR'])
    parser.add_argument('--weights_dir',type=str,default=configs['WEIGHTS_DIR'])
    parser.add_argument('--test_dir',type=str,default=configs['TEST_DIR'])
    parser.add_argument('--out_dir',type=str,default=configs['OUTPUT_DIR'])
    parser.add_argument('--encode_dir',type=str,default=configs['ENCODE_DIR'])

    parser.add_argument('--lr',type=float,default=configs['LR'])
    parser.add_argument('--epochs',type=int,default=configs['EPOCHS'])
    parser.add_argument('--batch_size',type=int,default=configs['BATCH_SIZE'])
    parser.add_argument('--imsize',type=int,default=configs['IMSIZE'])
    parser.add_argument('--category',type=str,default=configs['CATEGORY'])
    parser.add_argument('--sketch_views',type=str,nargs='*',default=configs['SKETCH_VIEWS'])
    parser.add_argument('-sketch_variations',type=int,default=configs['SKETCH_VARIATIONS'])
    parser.add_argument('--num_dn_views',type=int,default=configs['NUM_DN_VIEWS'])
    parser.add_argument('--num_dnfs_views',type=int,default=configs['NUM_DNFS_VIEWS'])
    

    return parser.parse_args()

global args_ 
args_ = parse_arguments() 