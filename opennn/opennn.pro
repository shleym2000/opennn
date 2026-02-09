#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = # Do not use qt

TARGET = opennn
TEMPLATE = lib

CONFIG += c++17

CONFIG += staticlib
CONFIG += precompile_header
CONFIG += jumbo_build

DEFINES += __Cpp17__

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:AVX2
    QMAKE_CXXFLAGS += /O3
}

win32-g++ {
    QMAKE_CXXFLAGS += -march=native
    QMAKE_CXXFLAGS += -mstackrealign
    QMAKE_CXXFLAGS += -O3
}

unix:!macx {
    QMAKE_CXXFLAGS += -march=native -O3
}

macx {
    QMAKE_CXXFLAGS += -march=native -O3
}

# Eigen library

INCLUDEPATH += ../eigen

# Source files

PRECOMPILED_HEADER = pch.h
HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

# CUDA

win32-msvc* | linux-g++ {
    CUDA_SOURCES += kernel.cu
}

include(../cuda.pri)

#OpenMP
include(../opennmp.pri)
