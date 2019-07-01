import os
import random
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from calibration import EntropyCalibrator
import common

import click

def build_int8_engine(model_file_path, calib, data, logger):
    with trt.Builder(logger) as builder, \
            builder.create_network() as network, \
            trt.UffParser() as parser:
        builder.max_batch_size = int(os.environ["BATCH_SIZE"])
        builder.max_workspace_size = common.GiB(2)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        
        parser.register_input(data["INPUT_NAME"], data["INPUT_SHAPE"])
        parser.register_output("Openpose/concat_stage7")
        parser.parse(model_file_path, network)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
    return engine


def load_random_batch(calib):
    batch = random.choice(calib.batch_files)
    _, data = calib.read_batch_file(batch)
    return data


@click.command()
@click.option("--dataset", default = "/home/akorovko/Code")
@click.option("--model_file", default = "/home/akorovko/Code/graph_opt.uff")
@click.option("--save_name", default = "openpose_int8.trt")
def main(dataset, model_file, save_name):
    calib = EntropyCalibrator(dataset)
    batch_size = calib.get_batch_size()

    TRT_LOGGER = trt.Logger()

    data = {
        "MODEL_FILE" : model_file,
        "INPUT_NAME" :"image",
        "INPUT_SHAPE" : (3, 368, 656)
    }

    with build_int8_engine(model_file, calib, data, TRT_LOGGER) as engine, \
        engine.create_execution_context() as context:
        
        with open(save_name, "wb") as f:
                f.write(engine.serialize())
    print("Done!")


if __name__ == '__main__':
    main()
