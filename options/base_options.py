import argparse
import os

import torch
import models
import data
import utils.utils as utils

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str,default="de_to_en",help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--project_name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
       
        parser.add_argument('--device', type=str, default=None)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        parser.add_argument('--num_encoder_options_each_layer', type=int, default=4, help='# of options in each layer')
        parser.add_argument('--num_decoder_options_each_layer', type=int, default=4, help='# of options in each layer')
        parser.add_argument('--num_encoder_shared_layers', type=int, default=0, help='# of options in each layer')
        parser.add_argument('--num_decoder_shared_layers', type=int, default=0, help='# of options in each layer')
        parser.add_argument('--num_classifier_encoder_layers', type=int, default=6, help='# of options in each layer')

        parser.add_argument("--src_vocab_size"         , type=int   , help="src vocab size"                                      , default=25000)
        parser.add_argument("--trg_vocab_size"         , type=int   , help="trg vocab size"                                      , default=25000)
        parser.add_argument("--model_dimension"        , type=int   , help="model dimmention"                                    , default=512)
        parser.add_argument("--number_of_heads"        , type=int   , help="number of heads"                                     , default=8)
        parser.add_argument("--dim_feedforward",type=int,default=2048)
        parser.add_argument("--num_encoder_layers"       , type=int   , help="number of layers"                                    , default=6)
        parser.add_argument("--num_decoder_layers", type=int, help="number of layers", default=6)

        parser.add_argument("--num_special_tokens_trg" , type=int   , help="number of special tokens of target"                  , default=2)
        parser.add_argument("--num_labels"             , type=int   , help="types of labels"                                     , default=6)
        parser.add_argument("--num_neg"                , type=int   , help="num of neg"                                          , default=2)
        parser.add_argument("--sequence_length"        , type=int   , help="sequence_length"                                     , default=128)
        
        parser.add_argument('--dataroot',type=str,default="./datasets", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        
        parser.add_argument("--tokenizer_path_src"     , type=str   , help="the saved tokenizer of src language"                 , default='{dataroot}/{name}/tokenizer.src')
        parser.add_argument("--tokenizer_path_trg"     , type=str   , help="the saved tokenizer of trg language"                 , default='{dataroot}/{name}/tokenizer.trg')
        
        parser.add_argument("--tokenizer_data_source_src"     , type=str   , help="the saved tokenizer of src language"                 , default='{dataroot}/{name}/train/de_to_en.de')
        parser.add_argument("--tokenizer_data_source_trg"     , type=str   , help="the saved tokenizer of src language"                 , default='{dataroot}/{name}/train/de_to_en.en')
        
        
        parser.add_argument("--data_path_train"    , type=str   , help="the training dataset of trg language"                , default='{dataroot}/{name}/train/')
        parser.add_argument("--data_path_eval"     , type=str   , help="the eval dataset of trg language"                    , default='{dataroot}/{name}/validation/')
       
        parser.add_argument("--model_save_path"         , type=str   , help="the directory to save and load model"                         , default='{checkpoints_dir}/{project_name}/{epoch}_{model_name}.pth')
        #parser.add_argument("--model_path_src"         , type=str   , help="the directory to load model"                         , default='./saved_models/saved_dict.pth')
        parser.add_argument("--step_losses_pth"        , type=str   , help="the path of the json file that saves step losses"    , default='{checkpoints_dir}/{project_name}/trails/{epoch}_step_losses.json')
        parser.add_argument("--train_losses_pth"       , type=str   , help="the path of the json file that saves train losses"   , default='{checkpoints_dir}/{project_name}/trails/{epoch}_train_losses.json')
        parser.add_argument("--eval_losses_pth"        , type=str   , help="the path of the json file that saves eval losses"    , default='{checkpoints_dir}/{project_name}/trails/{epoch}_eval_losses.json')
        parser.add_argument("--train_accuracy_pth"     , type=str   , help="the path of the json file that saves train accuracy" , default='{checkpoints_dir}/{project_name}/trails/{epoch}_train_accuracy.json')
        parser.add_argument("--eval_accuracy_pth"      , type=str   , help="the path of the json file that saves eval accuracy"  , default='{checkpoints_dir}/{project_name}/trails/{epoch}_eval_accuracy.json')
        parser.add_argument("--batch_max_len"         , type=int   , help="max length of the batch"  , default=10000)
        parser.add_argument("--batch_ignore_len"      , type=int   , help="ignore length of the batch"  , default=10000)
        parser.add_argument("--max_batch",type=int,default=float('inf'))
        parser.add_argument("--num_threads",type=int,default=4)
        parser.add_argument("--retokenize",type=bool,default=False)
        parser.add_argument("--save_in_memory",type=bool,default=False)
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        #print(opt.names)
        #for name in opt.names:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.project_name)
        
        #util.mkdirs(expr_dir)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
       
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test
        if opt.project_name is None:
            opt.project_name = opt.name
        if opt.device is None:
            opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            opt.device = torch.device(opt.device)
       

        self.print_options(opt)

        # set gpu ids
        #str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])
        # if isinstance( opt.names,str):
        #     opt.names=[opt.names]
        # if isinstance(opt.dataroot,str):
        #     opt.dataroot=[opt.dataroot]
            #print(torch.cuda.get_device_name(id))

        self.opt = opt
        return self.opt
