import logging
import inspect
import os
from pathlib import Path

from .writer import Writer
from .config import config_setup
from .dist import initialize as dist_init


def logger_setup(log_path, eval=False):

    log_fname = os.path.join(log_path, 'val.log' if eval else 'train.log')

    for hdlr in logging.root.handlers:
        logging.root.removeHandler(hdlr)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ],
    )
    main_filename, *_ = inspect.getframeinfo(inspect.currentframe().f_back.f_back)

    logger = logging.getLogger(Path(main_filename).name)
    writer = Writer(log_path)

    return logger, writer


def setup(args, extra_args=()):

    print('###########')
    distenv = dist_init(args)
    print('###########')

    args.model_config = Path(args.model_config).absolute().resolve().as_posix()

    if args.eval:
        config_path = Path(args.result_path).joinpath('config.yaml')
    elif args.resume:
        load_path = Path(args.load_path)
        if not load_path.is_file():
            raise ValueError("load_path must be a valid filename")

        config_path = load_path.parent.joinpath('config.yaml').absolute()
    else:
        config_path = Path(args.model_config).absolute()
        task_name = config_path.stem
        if args.postfix:
            task_name += f'__{args.postfix}'

    config = config_setup(args, distenv, config_path, extra_args=extra_args)

    return config
