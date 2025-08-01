#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=all
#SBATCH --mail-user=leo.dody1@univ-lyon3.fr
#SBATCH --output=pred.out #/dev/null
#SBATCH --job-name=pred
#SBATCH --partition=c6420-ib100


module purge
module use /easybuild/AlmaLinux/8/skylake-avx512/mlxln5.5/foss2022b/modules/all
module load Python/3.10.4-GCCcore-12.2.0
source /home_nfs/polytech/leo.dody/PhD/Article_2/PhD_article_2/.venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 ../src/models/regression_main.py -p True 

deactivate
