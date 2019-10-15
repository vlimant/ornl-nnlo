Some instructions to run over ORNL machines

# Titan

Setup the software

All necessary software is available with a local module
<pre>
module use /ccs/proj/csc291/vloncar/modules/modulefiles/
module load mpi_learn
</pre>

Run cifar10 example with bayesian.
<pre>
python2 scan-learn.py --masters 1:5 --hArgs 3:2:10 --hOp bayesian --force --model cifar10
</pre>

# Summit

# FlatIron Cluster (not ORNL)

Without Horovod support yet

<pre>
module load gcc
module load openmpi2
module load python3
module load python3-mpi4py
module load lib/hdf5/1.8.21-openmpi2
module load cuda/10.1.243_418.87.00
module load nccl
</pre>

Some user installed libs
<pre>
pip3 install --user scikit-optimize==0.5.2
</pre>
