#from keras.activations import relu, selu, elu
from keras.models import Model, Sequential
from keras.layers import Dense, Input, GRU, Dropout, Flatten

def get_model(**args):

    input_shape = (150,16)
    activation = ['relu', 'selu', 'elu']
    GRU_units=args.get('GRU_units',50)#300)
    DNN_neurons=args.get('DNN_neurons',100)#40)
    DNN_layers=args.get('DNN_layers',4)#2)
    DNN_activation_index=args.get('DNN_activation_index',2)#0)
    dropout=args.get('dropout',0.1)#0.2)

    inputArray = Input(shape=input_shape)
    x = GRU(GRU_units, activation="tanh",
            recurrent_activation='hard_sigmoid', name='gru')(inputArray)
    x = Dropout(dropout)(x)
    ####
    for i in range(0,DNN_layers):
        x = Dense(DNN_neurons, activation=activation[DNN_activation_index], 
                  kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
        x = Dropout(dropout)(x)
    #
    output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                   name = 'output_softmax')(x)
    ####
    model = Model(inputs=inputArray, outputs=output)
    #model.compile(optimizer=self.optimizer[self.optimizer_index], 
    #              loss='categorical_crossentropy', metrics=['acc'])
    return model

def get_name():
    return 'hls4ml-gru'

def get_all():
    import socket,os,glob
    host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))

    if 'daint' in host:
        all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/*.h5')
    elif 'titan' in host:
        all_list = glob.glob('/ccs/proj/csc291/DATA/hls-fml/NEWDATA/*_150p_*.h5')
    else:
        all_list = glob.glob('/bigdata/shared/hls-fml/NEWDATA/*_150p_*.h5')
    #self.X =  np.array(self.f.get('jetConstituentList'))
    #self.y = np.array(self.f.get('jets')[0:,-6:-1])
    return all_list

from skopt.space import Real, Integer, Categorical
get_model.parameter_range =     [
        Integer(200,300, name='GRU_units'),
        Integer(20,100, name='DNN_neurons'),
        Integer(1,5, name='DNN_layers'),
        Categorical([0,1,2], name='DNN_activation_index'),
        Real(0.0, 1.0, name='dropout')
]

def get_train():
    all_list = get_all()
    l = int( len(all_list)*0.70)
    train_list = all_list[:l]
    return train_list

def get_val():
    all_list = get_all()
    l = int( len(all_list)*0.70)
    val_list = all_list[l:]
    return val_list

def get_features():
    return 'jetConstituentList'

def get_labels():
    return 'jet_target'
