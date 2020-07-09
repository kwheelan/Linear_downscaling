#!/bin/bash -l
#SBATCH --job-name=OLS
#SBATCH --account=p48500028
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=out.job.%j
#SBATCH --mem=300G

#A script to downscale linearly using all passed predictors

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

#where the script is located
cd /glade/work/kwheelan/Linear_downscaling/scripts

LAT=38.125
LON=-101.875
OBS=/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc
LOCATION=$TMPDIR
PREDS=pass

python regress.py $LAT $LON $OBS $LOCATION $PREDS
