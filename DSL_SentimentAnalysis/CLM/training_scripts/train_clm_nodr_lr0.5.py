from CLM import train

def log_with_print(log, context):
    print >>log, context
    print context


logfile = __file__ + 'log'
log = open(logfile, 'w')

round = 0
log_with_print(log, 'round ' + str(round) + 'begin ------------------------------- !!')
# change some for round


max_epochs = 100000

obj_directory = r'..\Sentiment_CLM_nodrop_lr0.5'
reload_model  = obj_directory + r'\T.npz'


train(round = round,
        saveto =            obj_directory + '\\round%d_model_lstm.npz'%(round),
        reload_model  =     reload_model,
        reload_option =     reload_model + '.pkl',
        dataset =           r'../data/imdb.pkl', #%(work_id + 1),
        encoder       =     'lstm',
        dropout_input =     None,
        dropout_output=     None,
        clip_c        =     5.,
        dim_word      =     500,
        dim_proj      =     1024,
        n_words       =     10000,
        #n_words_sqrt  =     n_words_sqrt,
        optimizer     =     'adadelta',
        lrate         =     0.5,
        maxlen        =     None,
        minlen        =     1,
        start_iter    =     0,
        start_epoch   =     0,
        max_epochs    =     max_epochs, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch_size    =     16,
        patience      =     100,
        validFreq     =     5000,
        saveFreq      =     50000000,
        dispFreq      =     1,
        sampleFreq    =     20000000,
        newDumpFreq   =     20000,
        syncFreq      =     5000000000,
        sampleNum     =     25,
        decay_c       =     0.,
        log           =     logfile,
        monitor_grad  =     False,
        sampleFileName=     obj_directory + '\\round%d_sample.txt'%(round),
        pad_sos = False,
        embedding = '../data/embedding500.npz'
        )

