# Residual-Channel-Attention-Network
Implementation of "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", ECCV 2018, [[arXiv]](https://arxiv.org/abs/1807.02758) 

## Train
1. Training Dataset is [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).The TFRecords for the same and npz files of the benchmark dataset can be found in the [drive link]().

2. Prepare the following Directory structure
--> benchmark
    --> npz files of benchmark dataset
--> code
    --> data
        --> TFRecords files
    --> All the code files
--> results
    --> set5 (similarly for set14,b100,urban100)
        --> gt
        --> predicted
--> eval
    --> All the evaluation scripts

3. cd to `code`. Specify the scale of upsampling by '--scale'. Default value is 4. 

4. Run ```python3 training.py ``` 

## Inference
Run ```python3 infer.py ``` . The output will be stored in `results`.

## Evaluation (NOTE : THIS REQUIRES MATLAB)
1. cd to eval

2. Run `./matlab_eval.sh`
