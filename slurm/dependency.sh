jid1=$(sbatch  run.cp2k.shadow.slurm | awk '{print $4}')
jid2=$(sbatch  --dependency=afterany:$jid1 run.cp2k.shadow.slurm | awk '{print $4}')
jid3=$(sbatch  --dependency=afterany:$jid2 run.cp2k.shadow.slurm | awk '{print $4}')
jid4=$(sbatch  --dependency=afterany:$jid3 run.cp2k.shadow.slurm | awk '{print $4}')
