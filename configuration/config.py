import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )

    # for baseline
    parser.add_argument("--recent_ratio", type=float, default=0.5, help="sampling ratio between recent and past")
    parser.add_argument("--cls_weight_decay", type=float, default=0.999)
    parser.add_argument("--weight_option", type=str, default="loss", help="weightsum softmax loss")
    parser.add_argument("--weight_ema_ratio", type=float, default=0.1, help="ema smoothing ratio of loss sum")


    # for twf
    parser.add_argument("--optim_wd", type=float, default=0, help="")
    parser.add_argument("--optim_mom", type=float, default=0, help="")
    parser.add_argument("--pre_epoch", type=int, default=1000, help="pre_train epoch")
    parser.add_argument("--samples_per_task", type=int, default=10000, help="explicit task boundary for twf")
    parser.add_argument("--sigma", type=int, default=10, help="Sigma of gaussian*100")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat period")
    parser.add_argument("--init_cls", type=int, default=100, help="Percentage of classes already present in first period")
    parser.add_argument("--min_resize_threshold", type=int, default=16, help="")
    parser.add_argument("--resize_maps", type=int, choices=[0,1], default=0, help="")

    # 얘네는 default가 없음
    parser.add_argument("--lambda_diverse_loss", type=float, help="")
    parser.add_argument("--lambda_fp_replay", type=float, help="")
    parser.add_argument("--lambda_fp", type=float, help="")
    parser.add_argument("--der_alpha", type=float, help="")
    parser.add_argument("--der_beta", type=float, help="")



    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )

    parser.add_argument(
        "--f_period", type=int, default=10000, help="Period for measuring forgetting"
    )

    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=['cutmix', 'randaug'],
        help="Additional train transforms [cutmix, cutout, randaug]",
    )

    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--use_kornia", type=bool, default=True, help="disable kornia")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    # GDumb
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1,
                        help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # RM & GDumb
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")

    # BiC
    parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")

    # AGEM
    parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    parser.add_argument('--beta', type=float, default=10.0, help='distillation strength')
    parser.add_argument('--ema_ratio', type=float, default=0.999, help='ema_ratio')
    parser.add_argument('--ema_ratio_2', type=float, default=0.998, help='ema_ratio_2')
    parser.add_argument('--cls_dim', type=int, default=10, help='Number of output dim reserved for each_class')
    parser.add_argument('--weighted', action="store_true", help='Use class-weighted distillation')
    parser.add_argument('--pred_based', action="store_true", help='Use pred based weighting')
    parser.add_argument('--trans_feature', action="store_true", help='Use train transformed sample for distillation')
    parser.add_argument('--feature_only', action="store_true", help='Use only the features of last layer')
    parser.add_argument('--loss_ema', type=float, default=0.999, help='ema_ratio for updating cls loss')
    parser.add_argument('--norm_loss', type=str, default='none', help='Use normalized cls and distill loss')
    parser.add_argument('--loss_ratio', type=str, default='none', help='Dynamic ratio strategy for cls and distill loss')

    parser.add_argument('--dma_mean', type=float, default=0.5, help='mean of dma weights, in period')
    parser.add_argument('--dma_var', type=float, default=0.75, help='variance ratio (var/mean^2) of dma weights')
    parser.add_argument('--fc_train', type=str, default='none', help='train mode of fc layer')
    parser.add_argument('--online_fc_mode', type=str, default='none', help='train mode of online fc layer')

    parser.add_argument('--reduce_bpdepth', action="store_true", help='Reduce backpropagation depth for distill loss')
    parser.add_argument('--importance', type=str, default='none', help='feature map importance type')
    parser.add_argument('--imp_ema', type=float, default=0.99, help='ema_ratio for updating importance')

    args = parser.parse_args()
    return args
