import os

config = """
[application]
enable-perf-measurement=0
perf-measurement-interval-sec=10000000
#gie-kitti-output-dir=streamscl

[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720
gpu-id=0

[source0]
enable=1
#Type - 1=CameraV4L2 2=URI 3=MultiURI
type=3
num-sources=1
uri={uri}
latency={latency}
gpu-id=0

[streammux]
gpu-id=0
batch-size={batch}
batched-push-timeout=1000
## Set muxer output width and height
width={width}
height={height}
cuda-memory-type=1

[sink0]
enable=1
#Type - 1=FakeSink 2=EglSink 3=File 4=RTSP Streaming
type=4
sync=0
source-id=0
gpu-id=0
codec=1
bitrate={bitrate}
cuda-memory-type=1
rtsp-port={outstream_port}

[osd]
enable=1
gpu-id=0
border-width=1
text-size=15
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Arial
show-clock=0
clock-x-offset=800
clock-y-offset=820
clock-text-size=12
clock-color=1;0;0;0

[primary-gie]
enable=1
gpu-id=0
batch-size={batch}
gie-unique-id=1
interval=0
labelfile-path=labels.txt
model-engine-file=/model/openpose_int8.trt
config-file=config_infer_primary_Openpose.txt
"""

config2 = """
[property]
gpu-id=0
net-scale-factor=1
#0=RGB, 1=BGR
model-color-format=0
model-engine-file=openpose_int8.trt
batch-size={batch}
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=1
num-detected-classes=21
interval=0
gie-unique-id=1
parse-func=0
is-classifier=0
output-blob-names=Openpose/concat_stage7
parse-bbox-func-name=NvDsInferParseCustomOpenPose
custom-lib-path=nvdsinfer_openpose/libnvdsinfer_openpose.so

[class-attrs-all]
threshold=0.5
#eps=0.1
#group-threshold=2
roi-top-offset=0
roi-bottom-offset=0
detected-min-w=0
detected-min-h=0
detected-max-w=0
detected-max-h=0
"""

if __name__ == "__main__":
    uri = os.environ["INPUT_URI"]
    batch = os.environ.get("BATCH_SIZE")
    
    bitrate = os.environ.get("BITRATE")
    bitrate = bitrate if bitrate is not None else 10000

    outstream_port = os.environ.get("OUTSTREAM_PORT")
    outstream_port = outstream_port if outstream_port is not None else 1234

    latency = os.environ.get("LATENCY")
    latency = latency if latency is not None else 200

    width = os.environ.get("OUTPUT_WIDTH")
    width = width if width is not None else 800

    height = os.environ.get("OUTPUT_HEIGHT")
    height = height if height is not None else 600

    with open("../openpose_config.txt", mode = "a") as f:
        f.write(
            config.format(
                uri=uri, 
                bitrate=bitrate, 
                outstream_port=outstream_port,
                batch=batch,
                width=width,
                height=height,
                latency=latency
                )
            )
    print(config.format(
                uri=uri, 
                bitrate=bitrate, 
                outstream_port=outstream_port,
                batch=batch,
                width=width,
                height=height,
                latency=latency
                ))

    with open("../config_infer_primary_Openpose.txt", mode = "a") as f:
        f.write(
            config2.format(
                uri=uri, 
                bitrate=bitrate, 
                outstream_port=outstream_port,
                batch=batch,
                width=width,
                height=height,
                latency=latency
                )
            )
    print(config2.format(
                uri=uri, 
                bitrate=bitrate, 
                outstream_port=outstream_port,
                batch=batch,
                width=width,
                height=height,
                latency=latency
                ))
    