import os
import sys
import optparse

parser = optparse.OptionParser()
parser.add_option('--model', default='cifar10')
parser.add_option('--mode', default=None)
parser.add_option('--checkp', default=None)
parser.add_option('--hOp', choices = ['genetic','bayesian'])
parser.add_option('--hArgs',default='5:3')
parser.add_option('--epochs', type=int, default=0)
parser.add_option('--torch',default=False, action="store_true")
parser.add_option('--force',action='store_true')
parser.add_option('--masters',default = '1:1,2,3,4,5,10,15,20,25,30,35,40,60,100#3:10,15,20#5:10,15,20',
                  help="Specify the master/worker schemes. #-separated schemes. master:worker_opt1,worker_opt2,...")
(options,args) = parser.parse_args()

label = options.model
backend='keras' if not options.torch else 'torch'
if label == 'gan':
    base_command = "python3 MPIGDriver.py dummy.json train_3d.list test_3d.list --loss dummy --master-gpu --features-name X --labels-name y --easgd  --worker-opt rmsprop"
    fixed_metric = None##"--target-metric 'val_acc,>,0.97' "
    #fixed_metric = "--early 'discriminator_model:classification_loss,~<,4'"
    n_epochs_def = 1
elif label == 'mnist':
    model = 'mnist_arch.json' if not options.torch else 'mnist_torch_arch.torch'
    base_command = "python3 TrainingDriver.py --model %s --train train_mnist.list --val test_mnist.list --loss categorical_crossentropy --master-gpu --backend %s "%( model , backend)
    #fixed_metric = "--target-metric 'val_acc,>,0.97' "
    fixed_metric = "--early-stop 'val_loss,~<,4' "
    n_epochs_def = 10
elif label =='cifar10':
    base_command = "python3 TrainingDriver.py --model cifar10_arch.json --train train_cifar10.list --val test_cifar10.list --loss categorical_crossentropy --master-gpu"
    fixed_metric = "--target-metric 'val_acc,>,0.95' "
    n_epochs_def = 5
elif label == 'topclass':
    model = 'topclass_arch.json' if not options.torch else 'topclass_torch_arch.torch'
    base_command = "python3 TrainingDriver.py --model %s --train train_topclass.list --val test_topclass.list --loss categorical_crossentropy --master-gpu --features-name Images --labels-name Labels --backend %s "%( model, backend)
    fixed_metric = "--target-metric 'val_acc,>,0.97' "
    n_epochs_def = 5
elif label =='hls4gru':
    base_command = "python3 TrainingDriver.py --model hls4mlGRU.py --loss categorical_crossentropy --master-gpu --features-name jetConstituentList --labels-name jet_target"
    fixed_metric = "--early-stop 'val_loss,~<,10' "
    n_epochs_def = 5
else:
    print ("not a good label", label)

if options.mode:
    base_command+= " --mode "+options.mode

wall_times = { 1 : 2,
               125 : 6 ,
               313 : 12,
               3749 : 24,
               11249 : 24
               }

n_masters = {}
for item in options.masters.split('#'):
    master,workers = item.split(':')
    n_masters[int(master)] = list(map(int,workers.split(',')))

for n_master,n_workers in n_masters.items():
    for n_worker in n_workers:
        n_nodes = n_worker+n_master
        ## edit the job name and label
        job_label = '{0}-{1}-{2}-{3}'.format( label, n_master,n_worker,backend.replace('-',''))
        if options.mode:
            job_label+='-{0}'.format( options.mode )
        if fixed_metric: 
            job_label+='-fixedmetric'
        n_epochs = 10000  if fixed_metric else n_epochs_def
        if options.epochs: n_epochs = options.epochs
        if not fixed_metric:
            job_label+='-{0}epochs'.format( n_epochs )

        if options.hOp:
            job_label+='-{0}Opt'.format(options.hOp)
            
        extra_com = ""# --monitor "
        s_check = '--checkpoint {0} --checkpoint-interval {1}'.format( job_label, options.checkp)
        if options.hOp:
            s_check+= ' --opt-restore '
            ## in case of hyper-parameter optimization
            n_fold,n_par,n_it = map(int, options.hArgs.split(':'))
            print ("for the Hopt, estimating on {0} folds of {1} nodes, running {2} parameter set in parrallel, ".format( n_fold, n_nodes, n_par))
            block_size = n_fold * n_nodes
            n_nodes = 1 + n_par * block_size
            job_label+='-{0}f-{1}p'.format( n_fold, n_par )
            print ("leading to a block size of {0}, using {1} nodes in total".format( block_size, n_nodes ))

            full_com ='{} --epochs {} {} {} --n-master {} --block-size {} --n-fold {} --num-iterations {} --hyper-opt {} {}'.format(
                'python3 OptimizationDriver.py --verbose --example {0}'.format( options.model ),
                n_epochs,
                fixed_metric if fixed_metric else "",
                s_check if options.checkp else "",
                n_master,
                block_size, 
                n_fold,
                n_it,
                options.hOp,
                extra_com
            )            
        else:
            s_check+= ' --restore {}'.format( job_label )
            full_com ='{0} --epochs {1} {2} --trial-name {3} {4} --n-masters {5} {6}'.format(
                base_command,
                n_epochs,
                fixed_metric if fixed_metric else "",
                job_label,
                s_check if options.checkp else "",
                n_master,
                extra_com
                )
        pbs_name = 'sub-{0}.pbs'.format(job_label)
        pbsid = '{0}.pbsid'.format( job_label)
        if os.path.isfile( pbs_name):
            print (pbs_name,"already created")
            if os.path.isfile(pbsid):
                print (pbsid,"already submitted")
                print (open(pbsid).read())
                if not options.force:
                    continue
            else:
                print ("no pbs id found")

        pbs = open( pbs_name,'w')


        last_node = None
        next_node = None
        for lim in sorted(wall_times.keys()):
            n = wall_times[lim]
            next_node = lim
            if n_nodes > lim: 
                n_hours = n
                last_node = lim
            else:
                break
        wall_time = "{0}:00:00".format( n_hours )


        #print (full_com)
        pbs.write("""#!/bin/bash
#    Begin PBS directives
#PBS -A csc291
#PBS -N {4}
#PBS -j oe
#PBS -l walltime={0}
#PBS -l nodes={1}
#PBS -l gres=atlas1%atlas2
#    End PBS directives and begin shell commands


###module load deeplearning
module use /ccs/proj/csc291/vloncar/modules/modulefiles/
module load mpi_learn

export HOME=/lustre/atlas/scratch/vlimant/csc291/homeDL/
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2

cd $HOME/{2}

## mpi-learn example with mnist : function just right
date
aprun -e PYTHONPATH=$PYTHONPATH -e HOME=$HOME -e TERM=xterm -N 1 -n {1} {3}
""".format( wall_time,
            n_nodes,
            'NNLO',
            full_com,
            job_label)
                  )
        pbs.close()
        os.system('tail -1 {0}'.format( pbs_name ))
        print ("a run time of",n_hours,"hours")
        print ("requesting",n_nodes,"nodes >",last_node,"switching to bin at >",next_node)
        if input("submit {0} ? ".format( pbs_name )) in ['y','yes']:
            jib = os.popen( 'qsub {0}'.format( pbs_name )).read()
            open(pbsid,'w').write( jib )
