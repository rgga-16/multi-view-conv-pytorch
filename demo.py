# Code for generating from user inputs here


import yaml
with open(r'./config.yaml') as file:
    configs = yaml.full_load(file)

    # for item,doc in configs.items():
    #     print(f'{item} : {doc}')
    print(configs['MODEL'])