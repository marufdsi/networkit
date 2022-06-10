#!/bin/bash


git pull

cmake -DNETWORKIT_BUILD_TESTS=ON -DNETWORKIT_NATIVE=ON -DCMAKE_CXX_COMPILER=icpc ..

make
