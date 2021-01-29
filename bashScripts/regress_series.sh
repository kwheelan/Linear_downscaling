#!/bin/bash -l
#SBATCH --job-name=OLS
#SBATCH --account=p48500028
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=out.job.%j
#SBATCH --mem=100G

#A script to downscale linearly using all passed predictors

module load python
ncar_pylib

#where the script is located
cd /glade/work/kwheelan/Linear_downscaling/pythonScripts

export PREDICTAND=tmin


export OBS=/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/$PREDICTAND.gridMET.NAM-22i.SGP.nc

for LAT in $(seq 32.125 2 38.125)
do
    for LON in $(seq -101.875 2 -93.875)
    do

      export TIME=historical
      export START=1976-01-01
      export END=2005-12-31
      export EXT=_19500101-20051231_dayavg_mpigrid.nc

      export TMPDIR=/glade/work/$USER/Linear_downscaling/GCM_downscaled/$TIME
      mkdir -p $TMPDIR

      python regress.py $LAT $LON

      export BETAS=$TMPDIR/$PREDICTAND\_lat$LAT\_lon$LON/betas
      export ROOT=/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/MPI-ESM-LR/mpigrid/$TIME/
      export SERIES=MPI-ESM-LR_$TIME\_r1i1p1_NAmerica

      python gen_time_series.py $LAT $LON $TMPDIR $BETAS $ROOT $EXT $SERIES $START $END

      #future
      export TIME=rcp85
      export START=2070-01-01
      export END=2099-12-31
      export EXT=_20060101-21001231_dayavg_mpigrid.nc

      export TMPDIR=/glade/work/$USER/Linear_downscaling/GCM_downscaled/$TIME
      mkdir -p $TMPDIR

      export ROOT=/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/MPI-ESM-LR/mpigrid/$TIME/
      export SERIES=MPI-ESM-LR_$TIME\_r1i1p1_NAmerica

      python gen_time_series.py $LAT $LON $TMPDIR $BETAS $ROOT $EXT $SERIES $START $END

    done
done
