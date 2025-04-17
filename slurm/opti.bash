#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=all
#SBATCH --mail-user=leo.dody1@univ-lyon3.fr
#SBATCH --output=ANN_model_%a.out #/dev/null #ANN_model_%a.out
#SBATCH --job-name=opti_model_%a
#SBATCH --partition=c6420-ib100
#SBATCH --array=0-9%10


module purge
module use /easybuild/AlmaLinux/8/skylake-avx512/mlxln5.5/foss2022b/modules/all
module load Python/3.10.4-GCCcore-12.2.0
source /home_nfs/polytech/leo.dody/PhD/Article_2/PhD_article_2/.venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 ../src/models/regression_main.py -ba True --job_id $SLURM_ARRAY_TASK_ID

deactivate
