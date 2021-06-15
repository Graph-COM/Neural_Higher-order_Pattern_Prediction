# Neural Higher-order Pattern(Motif) Prediction in Temporal Networks

# Requirements
* `python = 3.7`, `PyTorch = 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
matploblib==3.3.2
```
Refer to `environment.yml` for more details.

# Run the code
## Preprocess the dataset
#### Option 1: Use our data

Preprocess dataset
```{bash}
mkdir processed
python preprocess.py -d tags-ask-ubuntu
```

This code is to preprocess the dataset, including zero-padding the node features and edge features.

#### Option 2: Use your own data
First download the dataset, for example https://www.cs.cornell.edu/~arb/data/
```{bash}
unzip dataset.zip
mkdir processed
python preprocess.py -d <dataset>
```

Before running the code, check if ./log exists. If not, 

```{bash}
mkdir log
```

After preprocessing the dataset, we can run the code for three different questions.
## For Q1 type prediction
The task aims to solve the Q1 in the paper. What type of high-order interaction will most likely appear among u,v,w within (t, t + T_W]?

```{bash}
export OMP_NUM_THREADS=1
python main.py -d tags-ask-ubuntu
```

The output will be in the log file. We will both report the AUC and the confusion matrix.

## For Q2 time prediction
The task aims to solve the Q2 in the paper. For a triplet ({u, v}, w, t), given an interaction pattern in {Wedge, Triangle, Closure}, when will u, v, w first form such
a pattern?

```{bash}
export OMP_NUM_THREADS=1
mkdir time_prediction_output
python main.py -d tags-ask-ubuntu --time_prediction --time_prediction_type <time_prediction_type>
```
## Optional arguments
```{txt}
    --time_prediction_type For interpretation, we have 3 tasks. 1: Closure; 2: Triangle; 3: Wedge;  Default 0 means no time_prediction
```

The output will be in the ./time_prediction_output/ <dataset>_<time_prediction_type>.txt.
We report the NLL loss and MSE for training, validating, and testing sets.


## For Q3 interpretation
```{bash}
export OMP_NUM_THREADS=1
mkdir interpretation_output
python main.py -d tags-ask-ubuntu --interpretation --interpretation_type 1
```

## Optional arguments
```{txt}
    --interpretation_type: Interpretation type: For interpretation, we have 3 tasks. 1: Closure vs Triangle; 2: Closure + Triangle vs Wedge; 3: Wedge and Edge; Default 0 means no interpretation
```
The output will be in the ./interpretation_output/ <dataset>_<interpretation_type>.txt.
We report all the pattern we sample, with the times of each pattern appears in the first class and the total times it appears in both classes, and their ratio. We also report the mean score and variance of each pattern.


# Note

Finding edges, wedges, triangles, and closures process is in th utils.py. Since the finding process is time-consuming, the code will automatically save all the edges, wedges, triangles, closures in the saved_triplets. So the code doesn't need to run the process again in the future experiments.

# Results

Our model achieves the following performance on :

| Model name         |    tags-math-sx   | tags-ask-ubuntu |   congress-bills   |      DAWN       |  threads-ask-ubuntu  |
| ------------------ | ----------------- | --------------- | ------------------ |---------------- | -------------------- |
| HIT                |     77.05 ± 0.31    |   81.62 ± 0.69   |     81.10 ± 0.26    |   76.50 ± 0.79   |      86.25 ± 0.15     |

# Main code and functions
    
## preprocess.py
the preprocess of data. Transform the hypergraph to the graph.

## main.py 
the main code, including finding patterns and train/validate/test the model.

## train.py
the main training code.

## module.py
different modules, including the HIT class, classifiers, self-attention pooling, and sum pooling.

## utils.py
functions, including early stopper, ROC score function, preprocess the dataset, finding patterns and Dataset class.

## parser.py
parsers
