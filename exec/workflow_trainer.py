import sys

sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
# sys.path.append('/media/anant/dev/src/surgical_workflow/')

from argparse import ArgumentParser
# from exec.train_V1 import train
from exec.train_V5 import train
from utils.helpers import create_handlers


def main():

    parser = ArgumentParser()
    
    ## ====================== Optimizer  ====================== ##
    parser.add_argument("--optimizer", type=str, dest="optimizer",
                      help="Optimizer and its parameters", default="sgd")
    parser.add_argument("--momentum", type=float, dest="momentum",
                      help="Momentum for sgd", default=0.9)
    parser.add_argument("--wd", type=float, dest="weight_decay",
                      help="Weight decay", default=1e-4)
    parser.add_argument("--lr", type=float, dest="learning_rate",
                      help="Learning rate", default=1e-2)
    parser.add_argument("--device", type=str, dest="device",
                      help="Use the following device",
                      default='cuda:0')
    parser.add_argument("--loss", type=str, dest="loss_fn",
                      help="Loss function to use", default="ce")
    parser.add_argument("--epochs", type=int, dest="epochs",
                      help="Epochs", default=1)
    parser.add_argument("--validation_interval", type=int, dest="validation_interval",
                      help="Validation interval", default=-1)
    parser.add_argument("--validation_interval_epochs", type=int, dest="validation_interval_epochs",
                      help="Validation interval in epochs", default=-1)
    parser.add_argument("--max_iterations", type=int, dest="max_iterations",
                      help="Maximum training iterations", default=-1)
    parser.add_argument("--max_valid_iterations", type=int, dest="max_valid_iterations",
                      help="Maximum validation iterations", default=-1)
    parser.add_argument("--accumulate_count", type=int, dest="accumulate_count",
                      help="Loss accumulate count for optimizer", default=1)
    parser.add_argument("--earlystopping", type=str, dest="early_stopping", nargs='+',
                      help="Parameters of early stopping", default="")
    parser.add_argument("--half_precision", action="store_true", dest="use_half_precision",
                        help="Use half precision", default=False)
    parser.add_argument("--run_nfolds", type=int, dest="run_nfolds",
                      help="Stop after nfolds", default=-1)
    ## ====================== Optimizer  ====================== ##


    ## ====================== Scheduler  ====================== ##
    parser.add_argument("--scheduler", type=str, dest="scheduler",
                      help="Type of scheduler to use", default=None)
    parser.add_argument("--cycle_length", type=int, dest="cycle_length",
                      help="Cycle length of the restart loop", default=-1)
    parser.add_argument("--nrestarts", type=int, dest="n_restarts",
                      help="Number of restarts in a cycle", default=1)
    parser.add_argument("--n_param_updates", type=int, dest="n_param_updates",
                      help="Number of steps in each restart loop", default=1)
    parser.add_argument("--restart_factor", type=float, dest="restart_factor",
                      help="Factor to multiply with cycle length after restart", default=1.)
    parser.add_argument("--init_lr_factor", type=float, dest="init_lr_factor",
                      help="Factor to multiply with lr after restart", default=1.)
    parser.add_argument("--etamin", type=float, dest="eta_min",
                      help="Minimum lr for consine annealing scheduler", default=1e-6)
    ## UNSED
    parser.add_argument("--parameter", type=str, dest="parameter",
                      help="Parameter to scheduler", default="lr")
    parser.add_argument("--start_value", type=float, dest="start_val",
                      help="Start value for the parameter", default=0.)
    ## ====================== Scheduler  ====================== ##


    ## ====================== Data related ====================== ##
    parser.add_argument("-b", "--base", dest="base_path",
                      help="Data path will be split into train and validation", 
                      metavar="DIRECTORY")
    parser.add_argument("--imagesize", type=int, dest="image_size", nargs=2,
                      help="Resize image to this size", default=(320, 180))
    parser.add_argument("--threads", type=int, dest="n_threads",
                      help="Number of threads", default=4)
    parser.add_argument("--batchsize", type=int, dest="batch_size",
                      help="Batch size", default=32)
    parser.add_argument("--folds", type=int, dest="n_folds",
                      help="Number of folds", default=5)
    ## ====================== Data related ====================== ##



    ## ====================== Logger ====================== ##
    parser.add_argument("--vislogger", type=int, dest="vislogger_interval",
                      help="Interval for visual logger", default=10)
    parser.add_argument("--vislogger_env", type=str, dest="vislogger_env",
                      help="Name of the vislogger environment", default="Workflow")
    parser.add_argument("--vislogger_port", type=int, dest="vislogger_port",
                      help="Visdom port", default=8080)
    parser.add_argument("--printlogger", type=int, dest="printlogger_interval",
                      help="Interval for print logger", default=-1)
    parser.add_argument("--win_names", type=str, dest="vislogger_windows", nargs='+',
                      help="Window names", default=[''])
    parser.add_argument("--save_model", type=str, dest="save_model",
                      help="Initiate save model", default="")
    parser.add_argument("--suffix", type=str, dest="suffix",
                      help="Additional suffix for the results folder", default="")
    ## ====================== Data related ====================== ##

    ## ====================== Other handlers ====================== ##
    parser.add_argument("--pretrained", dest="pretrained", 
                      help="Path or name for pretrained model", metavar="FILE",
                      default=None)
    # parser.add_argument("--handlers", type=str, dest="handler_list", nargs='+',
    #                   help="List of handlers", default=[])
    # parser.add_argument("--step", type=float, dest="step_size",
    #                   help="Step size for the parameter", default=1.)
    # parser.add_argument("--checkpoint", type="bool", dest="model_checkpoint",
    #                   help="Use model checkpoint", default=False)
    # parser.add_argument("--checkpoint_interval", type="int", dest="checkpoint_interval",
    #                   help="Interval for model save", default=-1)
    # parser.add_argument("--checkpoint", type="bool", dest="model_checkpoint",
    #                   help="Use model checkpoint", default=False)
    # parser.add_argument("--printlogger", type="int", dest="printlogger_interval",
    #                   help="Interval for print logger", default=-1)
    
    ## ====================== Data related ====================== ##


    

    options = parser.parse_args()


    optimizer_ops = {'optimizer': options.optimizer,
                    'momentum': options.momentum,
                    'weight_decay': options.weight_decay,
                    'learning_rate': options.learning_rate,
                    'device': options.device,
                    'loss_fn': options.loss_fn,
                    'validation_interval': options.validation_interval,
                    'validation_interval_epochs': options.validation_interval_epochs,
                    'max_iterations': options.max_iterations,
                    'epochs': options.epochs,
                    'max_valid_iterations': options.max_valid_iterations,
                    'accumulate_count': options.accumulate_count,
                    'early_stopping': options.early_stopping,
                    'use_half_precision': options.use_half_precision,
                    'run_nfolds': options.run_nfolds}

    scheduler_ops = {'scheduler': options.scheduler,
                    'parameter': options.parameter,
                    'n_param_updates': options.n_param_updates,
                    #'end_val': options.end_val,
                    'cycle_length': options.cycle_length,
                    'n_restarts': options.n_restarts,
                    'restart_factor': options.restart_factor,
                    'init_lr_factor': options.init_lr_factor,
                    'eta_min': options.eta_min}


    data_ops = {'base_path': options.base_path,
                'image_size': options.image_size,
                'n_threads': options.n_threads,
                'batch_size': options.batch_size,
                'n_folds': options.n_folds}

    logger_ops = {'win_names': options.vislogger_windows,
                    'vislogger_interval': options.vislogger_interval,
                    'vislogger_env': options.vislogger_env,
                    'vislogger_port': options.vislogger_port,
                    'printlogger_interval': options.printlogger_interval,
                    'save_model': options.save_model,
                    'suffix':options.suffix}

    visualizer_ops = {}

    model_ops = {'pretrained': options.pretrained}





    # handler_ops = {'cycle_length': options.cycle_length,
    #                 'step_size': options.step_size,
    #                 'scheduler':}


    # print(options.handler_list)

    # handlers = create_handlers(options.handler_list, handler_ops,
    #                             optimizer_ops,
    #                             model_ops)

    train(optimizer_ops, data_ops, logger_ops, model_ops, scheduler_ops)
        # train_batch_size, val_batch_size, epochs, lr, momentum, vis_log_interval, print_log_interval, handlers)

    
    # parser.add_option("-v", "--valid", dest="valid_folder",
    #                   help="Validation path", metavar="DIRECTORY")
    # parser.add_option("-p", "--pretrained", dest="pretrained_model",
    #                   help="Pretrained model", metavar="FILE")
    # parser.add_option("--glr", action="store_true", dest="get_learning_rate",
    #                   help="Learning rate determination", default=False)
    # parser.add_option("--minlr", type="float", dest="minlr",
    #                   help="Minimum learning rate", default=-4)
    # parser.add_option("--maxlr", type="float", dest="maxlr",
    #                   help="Maximum learning rate", default=-1)
    # # parser.add_option("-s", "--save", action="store_true", dest="save_model",
    # #                   help="Save model", default=False)
    # parser.add_option("-e", "--epochs", type="int", dest="epochs",
    #                   help="Number of epochs", default=5)
    # parser.add_option("--maxiterations", type="int", dest="max_iterations",
    #                   help="Maximum number of iterations", default=1e6)
    # parser.add_option("--visualize", type="int", dest="visualize_after_iterations",
    #                   help="Visualize after iterations", default=20)
    # parser.add_option("--test", type="int", dest="test_after_iterations",
    #                   help="Test after iterations", default=1000)
    # parser.add_option("-s","--save", dest="save_model",
    #                   help="Save model", metavar="DIRECTORY")
    # parser.add_option("--lr", type="float", dest="learning_rate",
    #                   help="Learning rate", default=1e-4)
    
    # parser.add_option("--suffix", type="string", dest="suffix",
    #                   help="File name suffix or prefix", default="")
    # parser.add_option("--batchsize", type="int", dest="batch_size",
    #                   help="Batch size", default=16)
    # parser.add_option("--imagesize", type="int", dest="image_size",
    #                   help="Resize image to this size", default=320)
    # parser.add_option("--threads", type="int", dest="n_threads",
    #                   help="Number of threads", default=6)
    # parser.add_option("--patience", type="int", dest="patience",
    #                   help="How many steps to wait before early stop", default=10)
    # # parser.add_option("--validpatience", type="int", dest="valid_patience",
    # #                   help="How many steps to wait before early stop", default=10)
    # # parser.add_option("--trainpatience", type="int", dest="train_patience",
    # #                   help="How many steps to wait before increasing or decreasing the learning rate", default=10)
    # parser.add_option("--validiterations", type="int", dest="max_valid_iterations",
    #                   help="Maximum number of valid iterations", default=1e6)
    # parser.add_option("--use_scheduler", action="store_true", dest="use_scheduler",
    #                   help="Use learning rate scheduler", default=False)
    

if __name__ == "__main__":
    main()