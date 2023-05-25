import os
import subprocess as sp


datasets = ["Cora", "CiteSeer", "AmazonP", "AmazonC", "CoauthorC", "CoauthorP"]
"""
seeds = [0,1,2,3,4,39788]
epses = [0.5 ,1, 1.5, 2]
alphas = [50, 200, 600]
betas = [0.001, 0.01]
lambs = [0, 0.5, 1, 1.5, 2]

jobs = []
for dataset in datasets:
    for seed in seeds:
        for eps in epses:
            for alpha in alphas:
                for beta in betas:
                    for lamb in lambs:
                        log = "results/%s_%d_%g_%g_%g_%g"%(dataset, seed, eps, alpha, beta, lamb)
                        jobs.append({'dataset':dataset, 'seed': seed, 'eps': eps, 'alpha': alpha, 'beta': beta, 'lamb':lamb, 'log': log})
           
for job in jobs:
    print(job)

for job in jobs: 
    path = job['log']
    if not os.path.exists(path):
        sp.call(['mkdir', path])
        print("Starting: ", job)
        sp.call(['python', 'train.py',
            '--dataset', job['dataset'],
            '--seed', str(job['seed']),
            '--eps', str(job['eps']),
            '--alpha', str(job['alpha']),
            '--beta', str(job['beta']),
            '--lamb', str(job['lamb']),
            '--log', path
                ])
"""

jobs = []
for dataset in datasets:
    log = "results/%s"%dataset
    jobs.append({'dataset':dataset,'log': log})
    
for job in jobs: 
    path = job['log']
    if not os.path.exists(path):
        sp.call(['mkdir', path])
        print("Starting: ", job)
        sp.call(['python', 'train.py',
            '--dataset', job['dataset'],
            '--log', path
                ]) 