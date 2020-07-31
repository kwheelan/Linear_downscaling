#!/bin/bash -l
#SBATCH --job-name=OLS
#SBATCH --account=p48500028
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=out.job.%j
#SBATCH --mem=100G

#A script to downscale linearly using all passed predictors

export TMPDIR=/glade/scratch/$USER/downscaling_data
mkdir -p $TMPDIR

module load python
ncar_pylib

#where the script is located
cd /glade/work/kwheelan/Linear_downscaling/pythonScripts

OBS=/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc
export LOCATION=$TMPDIR
export PREDS=mslp

for LAT in $(seq 32.125 2 38.125)
do
    for LON in $(seq -101.875 2 -93.875)
    do
	      python regress.py $LAT $LON $OBS $LOCATION $PREDS #&
    done
done
