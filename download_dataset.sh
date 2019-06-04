#!/bin/bash
# data_loader automatically download the dataset if it is not found at 'root_dataset'
# If for some reason it is not found, you can use this script to do so
DIR=./data
OUT=Pascal_VOC
link=http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
TAR_FILE=VOCtrainval_11-May-2012.tar
mkdir -p $DIR
cd $DIR

if [ ! -f $TAR_FILE ]; then
  wget $link
fi

if [ ! -e $OUT ]; then
  mkdir $OUT
  tar -xvf $TAR_FILE -C $OUT
  rm -rf $TAR_FILE
fi

cd ..
