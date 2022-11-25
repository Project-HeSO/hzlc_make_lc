#!/bin/bash
# Before executing this, edit settings.py

CONDA_ENV="hzlc_clone"
NUMBER_OF_MPI_PROCESSES=20

conda activate $CONDA_ENV

dates=(20211006)

for e in ${dates[@]}; do
    python make_settings.py ${e}
    mpiexec -np $NUMBER_OF_MPI_PROCESSES python fits_analysis_debug.py 
    mpiexec -np $NUMBER_OF_MPI_PROCESSES python merge_rank_devided_files.py 
    mpiexec -np 1 python reshape_light_curve_per_fits_to_per_star.py 
    mpiexec -np 28 python pca.py    
    
done;





