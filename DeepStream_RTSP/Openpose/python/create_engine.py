import click
import tensorrt as trt

def GiB(val):
    return val * 1 << 30

class ModelData(object):
    MODEL_FILE = "graph_opt.uff"
    INPUT_NAME ="image"
    INPUT_SHAPE = (3, 368, 656)

def build_engine(model_file, logger, data):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = GiB(2)
        builder.max_batch_size = 10
        # Parse the Uff Network
        parser.register_input(data["INPUT_NAME"], data["INPUT_SHAPE"])
        parser.register_output("Openpose/concat_stage7")
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

@click.command()
@click.option("--model_path", default = "graph_opt.pb")
def main(model_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    data = {
        "MODEL_FILE": model_path,
        "INPUT_NAME": "image",
        "INPUT_SHAPE": (3, 368, 656),
    }
    
    engine = build_engine(model_path, TRT_LOGGER, data)
    
    serialized_engine = engine.serialize()
    with open("openpose.engine", 'wb') as f:
        f.write(serialized_engine)
        
if __name__ == "__main__":
    main()