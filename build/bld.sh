#!/bin/bash


git pull

cmake -DNETWORKIT_BUILD_TESTS=ON -DNETWORKIT_NATIVE=ON ..

make
