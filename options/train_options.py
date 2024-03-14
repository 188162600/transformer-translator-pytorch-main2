from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        #parser.add_argument("--num_options_each_layer"          , type=int   , help="number of epochs"                                    , default=20)
        parser.add_argument("--current_epoch"          , type=int   , help="current epoch"                                       , default=0)
        parser.add_argument("--num_of_epochs", type=int, help="current epoch", default=40)

        parser.add_argument("--optimize_C_epoch", type=int,default=0)
        parser.add_argument("--optimize_G_epoch", type=int,default=0)
        #parser.add_argument("--batch_size"             , type=int   , help="batch size"                                          , default=5)
        parser.add_argument("--learning_rate"          , type=float , help="learning rate"                                       , default=1e-4)
        parser.add_argument("--weight_decay"           , type=float , help="weight_decay"                                        , default=1e-4)
        parser.add_argument("--dropout_probability"    , type=float , help="dropout prob"                                        , default=1e-1)
        parser.add_argument("--model_load_path"        , type=str   , help="the directory to load model"                         , default=None)
        parser.add_argument("--save_epoch_freq", type=int,default=1)
        #parser.add_argument("--batch_size", type=int, help="batch size", default=5)
        self.is_train = True
        return parser
    def parse(self):
        opt=super().parse()
        # if len(opt.optimize_G_epoch)==0:
        #     opt.optimize_G_epoch=[0]*len(opt.names)
       
        # if len(opt.optimize_C_epoch)==0:
        #     opt.optimize_C_epoch=opt.optimize_G_epoch
        # opt.load_data_epoch=[]
        # for optimize_G_epoch,optimize_C_epoch in zip(opt.optimize_G_epoch,opt.optimize_C_epoch):
        #     opt.load_data_epoch.append(min(optimize_G_epoch,optimize_C_epoch))
        opt.continue_train=opt.model_load_path is not None
        
        return opt
