import os, sys, math

template = open('t_submit_t').read()

tests=[
    #{'convergence': '--epochs 1',
    # 'convergence_l' :'',
    # 'gpu' : 0},
    {'convergence': '--epochs 1',
     'convergence_l' :'1epoc',
     'gpu' : 1},

    #{'convergence': '--epochs 10000 --early "val_loss,~<,4"',
    # 'convergence_l' :'early',
    # 'gpu' : 0},
    {'convergence': '--epochs 10000 --early "val_loss,~<,4"',
     'convergence_l' :'early',
     'gpu' : 1},

    #{'convergence': '--epochs 10000 --target-metric "val_acc,>,0.90"',
    # 'convergence_l' :'0.9',
    # 'gpu' : 0},
    #{'convergence': '--epochs 10000 --target-metric "val_acc,>,0.90"',
    # 'convergence_l' :'0.9',
    # 'gpu' : 1},
]


DO = (len(sys.argv)>1 and sys.argv[1]=='1')

cache='--cache /imdata/'
mpi_command = 'mpirun -x TERM=linux --map-by node --hostfile hostf --prefix /opt/openmpi-3.1.0 -np {np} --tag-output '
singularity_exe = 'singularity exec --nv -B /imdata/ -B /storage/ /storage/group/gpu/software/singularity/ibanks/edge.simg'
##model_run = 'python3 TrainingDriver.py --model cifar10_arch.json --train train_cifar10.list  --val test_cifar10.list --loss categorical_crossentropy {condition}  {mgpu} --n-process {nhrv} --batch {bsize} --trial-name {tname}'
bare_bsize = 100
model_run = 'python3 TrainingDriver.py --model examples/example_jedi_torch.py --loss categorical_crossentropy {condition}  {mgpu} --n-process {nhrv} --batch {bsize} --trial-name {tname} --checkpoint-interval 5 --checkpoint {tname}'
batch_scale = True

base_command = ' '.join( [
    #mpi_command,
    #singularity_exe,
    model_run,
    #cache
])



for test in tests:
    print('-'*40)
    #print(test)
    name=""
    convergence = test.get('convergence')
    name+= test.get('convergence_l')
    gpu = test.get('gpu')
    if gpu==0: name+='cpu'


    for (W,P) in [
            (0,1), 
            (1,1), 
            (2,1), 
            (3,1), (3,2), 
            (4,1), (4,3), 
            (5,2), (5,4),
            (10,1), (10,2), (10,3), (10,4),
            (15,1), (15,2), (15,3), (15,4),
            (20,1), (10,2), (20,3), (20,4)
    ]:
        if P!=1: continue
        #print('-'*40)
        tname='jedi_{0}_{1}_{2}'.format(W, P, name)
        bsize = bare_bsize*P if batch_scale else bare_bsize
        if batch_scale: tname+='_BS{}'.format( bsize )
        #print(tname)
        #if os.path.isfile('cifar10_arch_{}_history.json'.format( tname )):
        #    print('\t'*2,tname,'already done')
        #    continue

        mgpu = '--max-gpu {}'.format( gpu ) if gpu==0 else ''

        command = base_command.format( np = W,
                                       nhrv = P,
                                       condition = convergence,
                                       mgpu = mgpu,
                                       tname = tname,
                                       bsize = bsize
                         )
        ntasks = 1+W*P
        nnodes = int(math.ceil(ntasks/5.))
        print ntasks,"on",nnodes
        print "\t",command
        #if DO: os.system(command)
        
        sname = tname+'.sh'
        open(sname,'w').write( template.format( ntasks = ntasks,
                                                      nnodes = nnodes,
                                                      python_command = command
                                                  ))
                                                      
        if DO: os.system('sbatch {}'.format( sname ))
