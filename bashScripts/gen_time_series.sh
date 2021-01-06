#!/bin/bash -l
#SBATCH --job-name=OLS
#SBATCH --account=p48500028
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=out.job.%j
#SBATCH --mem=100G

#A script to downscale linearly using all passed predictors

export TMPDIR=/glade/scratch/$USER/downscaling_data/predictions
mkdir -p $TMPDIR

module load python
ncar_pylib

#where the script is located
cd /glade/work/kwheelan/Linear_downscaling/pythonScripts

export LAT=32.125
export LON=-101.875
export BETAS=/glade/scratch/kwheelan/downscaling_data/metrics/tmax_lat32.125_lon-101.875/betas/betas_tmax_32.125_-101.875.txt
export ROOT=/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/MPI-ESM-LR/mpigrid/
export EXT='19500101-20051231_dayavg_mpigrid.nc'
export SERIES='MPI-ESM-LR_historical_r1i1p1_NAmerica'

python gen_time_series.py $LAT $LON $TMPDIR $BETAS $ROOT $EXT $SERIES
