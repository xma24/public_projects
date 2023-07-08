import logging
import os
import time

import torch.distributed as dist


class Utils:
    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
        logger_initialized = {}

        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        # handle hierarchical names
        # e.g., logger "a" is initialized, then logger "a.b" will skip the
        # initialization since it is a child of "a".
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger

        # handle duplicate logs to the console
        # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
        # to the root logger. As logger.propagate is True by default, this root
        # level handler causes logging messages from rank>0 processes to
        # unexpectedly show up on the console, creating much unwanted clutter.
        # To fix this issue, we set the root logger's StreamHandler, if any, to log
        # at the ERROR level.
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # only rank 0 will add a FileHandler
        if rank == 0 and log_file is not None:
            # Here, the default behaviour of the official logger is 'a'. Thus, we
            # provide an interface to change the file mode to the default
            # behaviour.
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        if rank == 0:
            logger.setLevel(log_level)
        else:
            logger.setLevel(logging.ERROR)

        logger_initialized[name] = True

        return logger

    @staticmethod
    def get_root_logger(log_file=None, log_level=logging.INFO):
        logger = Utils.get_logger(
            name="DeepModel", log_file=log_file, log_level=log_level
        )

        return logger

    @staticmethod
    def console_logger_start():
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        os.makedirs("./work_dirs/", exist_ok=True)
        console_log_file = os.path.join("./work_dirs/", f"{timestamp}.log")
        console_logger = Utils.get_root_logger(
            log_file=console_log_file, log_level=logging.INFO
        )
        return console_logger
