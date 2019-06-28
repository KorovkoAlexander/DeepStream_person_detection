CUDA_VER:=10.0
CC:= g++

CFLAGS:= -Wall -Werror -std=c++11 -shared -fPIC
CFLAGS+= -I/root/DeepStream_Release/sources/includes -I/usr/local/cuda/include -I/nvidia/TensorRT-5.1.2.2/include

OPENCV = `pkg-config opencv --cflags --libs`

LIBS:= $(OPENCV) -L/nvidia/TensorRT-5.1.2.2/lib -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsparsebbox_Openpose.cpp
TARGET_LIB:= libnvdsinfer_openpose.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
