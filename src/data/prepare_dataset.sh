#!/bin/bash
rawdatadir=../../data/raw/grasps/
interdatadir=../../data/interim/grasp/
mkdir -p $interdatadir
cd $rawdatadir
for f in *.tar.gz; do tar -xvf "$f" -C "../../interim/grasp/"; done

