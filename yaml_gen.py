import yaml

setup = {'data_options': {'CGM': 3, 'CHO': 0, 'Insulin': 0}, 
         'output_window' : 6,
         'stride' : 1, 
         'LSTM': [7, 5, 5],
         'FC': [3, 3],
         'dropout' : 0.5,
         'epoch' : 1,
         'batch' : 256,
         'learning_rate' : 1e-4,
         'SEED' : 1,
         'num_layers' : [1, 1, 1]
         #'decay' : 
         }

f = open("Settings.yaml", 'w')
yaml.dump(setup, f, allow_unicode=True)