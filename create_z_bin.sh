#!/bin/bash

DATADIR=dataset/cnndm/processed
OUTPUTDIR=dataset/cnndm/processed_bin
mkdir -p $OUTPUTDIR

bash z_bin.sh $DATADIR $OUTPUTDIR
