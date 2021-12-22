#!/bin/bash

# SLURM pragmas (#SBATCH ...)
# These pragmas are camouflaged as shell comments! 
# They will be executed, even though prefaced with #
# To disable a #SBATCH pragma double comment it (##SBATCH ...) or add a space (# SBATCH ...)
# Lots of information: man sbatch

# Useful escapes:
#       %x : Will be replaced by the jobname (name given by SBATCH -J <JobName>)
#       %j : Will be replaced by the actual SLURM job id
#       $SLURM_JOB_ID : 

# To submit the job described in this file do
#       sbatch <ThisFile> : Submits job and returns <JobID>

# Other interesting SLURM commands
#       squeue : Lists all your jobs
#       sjobs <JobID> : Print detailed information about job <JobID>
#       scancel <JobID> : Kill specified job
#       scancel -u <TUID> : Kill all your own jobs
#       csreport : shows information about used computing time per project
#       csum / seff <JobID> / tuda-seff <JobID> : shows computing time and resource usage (efficiency)

# Personalized Settings:
# TUID is used to:
#       - Places stdout/stderr files in your /work/scratch/<TUID>/ folder
#       - Set the initial directory for the job to /work/scratch/<TUID>/
# Please replace it with your own (until we have a group/project space)


# --- REQUIRED PARAMETERS ---

# Number of tasks (=processes) the job uses. 
# Multiple processes can always be dispatched to different nodes = MPI is required
# Therefore set n to 1
#SBATCH -n 1

# Number of CPU cores per process in case of multi-threading
# From 1 to 96, all threads run on the same node
# Common values: 24 | 48 | 96
#SBATCH -c 96

# Maximum memory per core in MBytes
#SBATCH --mem-per-cpu 2048

# Time limit in wall clock time
# Different categories:
#       <= 30m : Short jobs that can run almost everywhere and some nodes are reserved for short jobs
#       <= 24h : Default job can run on almost all nodes
#       >  24h : Run only on select nodes, max is 7d
# Categories probably irrelevant when requesting GPUs, but still ... 
#SBATCH -t 00:15:00

# Partition to use, possible values : see sinfo
# I could not find an explanation but they seem to align with the job categories?
# Not needed anymore?
##SBATCH -p acc_short

# --- RECOMMENDED PARAMETERS ---

# Project name: 'project' + 5 digits (left pad 0)
#SBATCH -A project01728

# Name of the job, just for the humans for easier reading, does not have to be unique
#SBATCH -J MOTIS-GPU-VOLTA

# Write jobs stdout to the given file
#SBATCH -o /work/scratch/ls98semu/%x.stdout.%j

# Write the jobs stderr to the given file
#SBATCH -e /work/scratch/ls98semu/%x.stderr.%j

# Send a mail on job start and on job end
#SBATCH --mail -user=leon.steiner@stud.tu-darmstadt.de
#SBATCH --mail -type=ALL

# Working directory for the commands below (like 'cd some/where')
#SBATCH -D /work/scratch/ls98semu/

# Requests that the node the job is running on is exclusive to this job 
# (i.e. no other jobs at the same time on this node)
# We need this for benchmarking, otherwise try not using this and comment out
#SBATCH --exclusive

# Select nodes given certain (C)onstraints or (g)eneric (res)ources
# You can chain constraints with & and | (e.g. -C "avx512&mem1536"
# Not every possible combination of constraints is available
#
#       -C "avx512" : Use nodes with AVX512 capable CPUs
#       -C "mem1536" : Lots of memory
#       -C "mpi" : Use MPI
#       --gres=gpu : Any NVIDIA GPU
#       --gres=gpu:a100 : NVIDIA Ampere 100
#       --gres=gpu:v100 : NVIDIA Volta 100
#
# You can specify the amount of generic resources after requesting it:
# --gres-gpu:a100:2 : Request two A100 GPUs
# Max is currently 4, as there are only nodes with at most 4 GPUs
#SBATCH --gres=gpu:v100:1

# Submit a job array. Use this if you have multiple jobs to queue
##SBATCH -a

# --- ACTUAL JOB PAYLOAD --- #

module purge
module load gcc/10.2.0 cuda/11.5

srun ./motis --batch_output_file=/work/scratch/ls98semu/10k-ontrip-responses-raptor_gpu-v100.txt
# Save exist status from the actual run for later
EXITSTATUS=$?

echo "Job $SLURM_JOB_ID with name $SLURM_JOB_NAME on account $SLURM_JOB_ACCOUNT has finished at $(date)."

exit $EXITSTATUS
