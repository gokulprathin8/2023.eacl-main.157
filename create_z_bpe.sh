#!/bin/bash
FILE=cnndm
DATADIR=dataset/cnndm
OUTPUTDIR=dataset/cnndm/processed
mkdir -p $OUTPUTDIR

for _type in test train val; do
    for _file in source target; do
        bash z_bpe.sh $DATADIR/$_type.$_file $OUTPUTDIR/$_type.bpe.$_file
    done
done

