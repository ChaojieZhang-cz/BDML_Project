#!/bin/bash

#SBATCH --job-name=download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=100:00:00

module purge
cd /scratch/cz2064/myjupyter/BDML/Project/data
source /scratch/cz2064/gdc_client/gdc-client-develop/venv/bin/activate
gdc-client download -m /scratch/cz2064/myjupyter/BDML/Project/gdc_manifest_20200325_231804.txt
