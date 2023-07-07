import logging
import os
import time

import torch
import torch.distributed as dist
from c2_data import Data


class Utils:
    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
        logger_initialized = {}

        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0 and log_file is not None:
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

    @staticmethod
    def get_max_batchsize(data_dir, model):
        # Define the initial batch size and the maximum memory utilization
        max_memory_utilization = 0.9 * (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )  # Use up to 90% of available GPU memory
        print(f"==>> max_memory_utilization: {max_memory_utilization}")

        # Initialize the batch size and memory usage
        batch_size = 1
        memory_usage = 0

        model.to("cuda")

        count = 0
        memeory_usage_dict = {}
        # Iterate until memory usage exceeds the threshold
        while memory_usage < max_memory_utilization:
            # print(f"==>> memory_usage: {memory_usage}")
            # Increase the batch size
            # if memory_usage < max_memory_utilization * 0.5:
            #     batch_size *= 2
            # else:
            #     batch_size += 1
            # print(f"==>> batch_size: {batch_size}")

            if count == 2:
                memory_ratio_int = int(
                    (max_memory_utilization - memeory_usage_dict[0])
                    // (memeory_usage_dict[1] - memeory_usage_dict[0])
                )
                batch_size *= memory_ratio_int
                break

            # elif count > 1:
            #     batch_size += 1

            batch_size = int(batch_size)
            print(f"==>> batch_size: {batch_size}")

            dataloader = Data.get_dataloader(
                batch_size, data_dir, data_category="train"
            )

            # Iterate over a small number of batches to measure memory usage
            for images, labels in dataloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                _ = model(images)

                memory_usage = torch.cuda.max_memory_allocated() / 1024**3
                print(f"==>> memory_usage: {memory_usage}")

                memeory_usage_dict[count] = memory_usage

                images = images.to("cpu")
                labels = labels.to("cpu")
                del images
                del labels
                break
            count += 1
        del model

        print(f"Selected batch size: {batch_size}")

        return batch_size
