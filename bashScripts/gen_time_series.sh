#!/bin/bash -l
#SBATCH --job-name=OLS
#SBATCH --account=p48500028
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:02:00
#SBATCH --output=out.job.%j
#SBATCH --mem=100G

#A script to downscale linearly using all passed predictors

module load python
ncar_pylib

#where the script is located
cd /glade/work/kwheelan/Linear_downscaling/pythonScripts

export TMPDIR=/glade/scratch/$USER/downscaling_data/predictions
mkdir -p $TMPDIR

#export TIME=historical
#export START=1976-01-01
#export END=2005-12-31
#export EXT=_19500101-20051231_dayavg_mpigrid.nc

export TIME=rcp85
export START=2070-01-01
export END=2099-12-31
export EXT=_20060101-21001231_dayavg_mpigrid.nc

export LAT=32.125
export LON=-101.875
export BETAS=/glade/scratch/kwheelan/downscaling_data/metrics/tmax_lat32.125_lon-101.875/betas
export ROOT=/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/MPI-ESM-LR/mpigrid/$TIME/
export SERIES=MPI-ESM-LR_$TIME\_r1i1p1_NAmerica

python gen_time_series.py $LAT $LON $TMPDIR $BETAS $ROOT $EXT $SERIES $START $END
