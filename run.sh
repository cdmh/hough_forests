#!/bin/sh
export LD_LIBRARY_PATH=/usr/pack/opencv-1.0.0-dr/amd64-debian-linux4.0/lib:$LD_LIBRARY_PATH
exec ./CRForest-Detector $*
