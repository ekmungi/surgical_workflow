import sys

# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/endovis/')

from argparse import ArgumentParser
from workflow.exec.train_V3 import train
from workflow.utils.helpers import create_handlers


def main():

    parser = ArgumentParser()
    
    ## ====================== Optimizer  ====================== ##
    parser.add_argument("--optimizer", type=str, dest="optimizer",
                      help="Optimizer and its parameters", default="sgd")
    parser.add_argument("--momentum", type=float, dest="momentum",
                      help="Momentum for sgd", default=0.9)
    parser.add_argument("--wd", type=float, dest="weight_decay",
                      help="Weight decay for sgd", default=1e-4)
    parser.add_argument("--lr", type=float, dest="learning_rate",
                      help="Learning rate", default=1e-2)
    parser.add_argument("--device", type=str, dest="device",
                      help="Use the following device",
                      default='cuda:0')
    parser.add_argument("--loss", type=str, dest="loss_fn",
                      help="Loss function to use", default="ce")
    ## ====================== Optimizer  ====================== ##


    # ## ====================== Scheduler  ====================== ##
    # parser.add_argument("--scheduler", type="string", dest="scheduler",
    #                   help="Type of scheduler to use", default=None)
    # parser.add_argument("--parameter", type="string", dest="parameter",
    #                   help="Parameter to scheduler", default="lr")
    # parser.add_argument("--start_value", type="float", dest="start_val",
    #                   help="Start value for the parameter", default=0.)
    # parser.add_argument("--end_value", type="float", dest="end_val",
    #                   help="End value for the parameter", default=1.)
    # ## ====================== Scheduler  ====================== ##


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
    parser.add_argument("--epochs", type=int, dest="epochs",
                      help="Epochs", default=1)
    parser.add_argument("--folds", type=int, dest="n_folds",
                      help="Number of folds", default=5)
    parser.add_argument("--validation_interval", type=int, dest="validation_interval",
                      help="Validation interval", default=1000)
    parser.add_argument("--max_iterations", type=int, dest="max_iterations",
                      help="Batch size", default=5000)
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
    ## ====================== Data related ====================== ##

    ## ====================== Other handlers ====================== ##
    parser.add_argument("--pt", type=str, dest="pretrained",
                      help="Path or name for pretrained model",
                      default=None)
    parser.add_argument("--handlers", type=str, dest="handler_list", nargs='+',
                      help="List of handlers", default=[])
    parser.add_argument("--cycle_length", type=int, dest="cycle_length",
                      help="Cycle length of the restart loop", default=1500)
    parser.add_argument("--step", type=float, dest="step_size",
                      help="Step size for the parameter", default=1.)
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
                    'max_iterations': options.max_iterations}

    # scheduler_ops = {'scheduler': options.scheduler,
    #                  'parameter': options.parameter,
    #                  'start_val': options.start_val,
    #                  'end_val': options.end_val}


    data_ops = {'base_path': options.base_path,
                'image_size': options.image_size,
                'n_threads': options.n_threads,
                'batch_size': options.batch_size,
                'n_folds': options.n_folds,
                'epochs': options.epochs}

    logger_ops = {'vislogger_interval': options.vislogger_interval,
                    'vislogger_env': options.vislogger_env,
                    'vislogger_port': options.vislogger_port,
                    'printlogger_interval': options.printlogger_interval}

    model_ops = {'pretrained': options.pretrained}

    handler_ops = {'cycle_length': options.cycle_length,
                    'step_size': options.step_size}

    handlers = create_handlers(options.handler_list, handler_ops,
                                optimizer_ops,
                                model_ops)

    train(optimizer_ops, data_ops, logger_ops, model_ops, handlers)

    

if __name__ == "__main__":
    main()