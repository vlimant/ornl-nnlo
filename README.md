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
