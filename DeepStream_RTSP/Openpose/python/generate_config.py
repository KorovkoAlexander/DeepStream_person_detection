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
gpu-id=0

[streammux]
gpu-id=0
batch-size=10
batched-push-timeout=1000
## Set muxer output width and height
width=1280
height=720
cuda-memory-type=1

[sink0]
enable=1
#Type - 1=FakeSink 2=EglSink 3=File 4=RTSP Streaming
type=4
sync=1
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
batch-size=10
gie-unique-id=1
interval=0
labelfile-path=labels.txt
model-engine-file=/model/openpose_int8.trt
config-file=config_infer_primary_Openpose.txt
"""

if __name__ == "__main__":
    uri = os.environ["INPUT_URI"]
    
    bitrate = os.environ.get("BITRATE")
    bitrate = bitrate if bitrate is not None else 10000

    outstream_port = os.environ.get("OUTSTREAM_PORT")
    outstream_port = outstream_port if outstream_port is not None else 1234

    with open("../openpose_config.txt", mode = "a") as f:
        f.write(config.format(uri=uri, bitrate=bitrate, outstream_port=outstream_port))
    print(config.format(uri=uri, bitrate=bitrate, outstream_port=outstream_port))
    