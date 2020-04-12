#!/bin/bash

srun --pty --cpus-per-task=1 --mem-per-cpu=6100mb --partition=cpsc424 -t 6:00:00 bash
module load miniconda
module load FFmpeg/3.4.2-foss-2018a
source activate score_following