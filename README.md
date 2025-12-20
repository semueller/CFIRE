CFIRE Pruning experiments

`cfire_requirements` contains conda requirements;
`requirements.txt` contains pip requirements

Executing the following produces the results, figures and tables for submission.
```shell
cd pruning_experiments
python experiments_pruning.py 
python eval_esann.py
```

Independent from the experiments, `pruning_experiments/standalone_example.py` contains an example where 
- a black box is trained, 
- a CFIRE model is computed using integrated_gradients and
- threshold_pruning is applied with a visualization of Test-Accuracy, Ambiguity and Size for various pruning thresholds

(Note that some datasets may need to be sourced manually)