# change paths if necessary
INCLUDES = -I/usr/pack/opencv-1.0.0-dr/amd64-debian-linux4.0/include/opencv
LIBS = -lcxcore -lcv -lcvaux -lhighgui -lml
LIBDIRS = -L/usr/pack/opencv-1.0.0-dr/amd64-debian-linux4.0/lib

OPT = -O3 -Wno-deprecated

CC=g++

.PHONY: all clean

OBJS = CRForest-Detector.o CRPatch.o HoG.o CRForestDetector.o CRTree.o

clean:
		rm -f *.o *~ CRForest-Detector
			
all:	CRForest-Detector
		echo all: make complete
%.o:%.cpp
	$(CC) -c $(INCLUDES) $+ $(OPT)
	
CRForest-Detector: $(OBJS)
		$(CC) $(LIBDIRS) $(LIBS) -o $@ $+ $(OPT)
 




