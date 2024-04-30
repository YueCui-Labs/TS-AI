#! /bin/bash


method=$1

sbatch 01_test_retest.sh $method
sbatch 02_beh_reg.sh $method
sbatch 03_dr_homo.sh $method
sbatch 04_task_homo.sh $method

