# Detect Box and Predict VIN-Number using Tensorflow
This repo is for Detecting box and to Predict VIN-Number based on 'Neural Network' methods. Use CNN Models and Use OpenCV package for detection.  

The project folder structure is as follows.  
```bash
❯ tree -N -L 3
.
├── README.md
├── cnn-example
│   ├── cnn-example.py
│   └── cnn-summary.md
└── predict-digit
    ├── blackbox.py
    ├── core
    │   ├── common.py
    │   └── model.py
    ├── main.py
    ├── preprocessing.py
    ├── res
    │   ├── box
    │   └── char
    └── tmp
        ├── Fnt.zip
        ├── box
        ├── sample1.jpg
        ├── sample2.jpg
        └── sample3.jpg

8 directories, 12 files
```

## Requirements
The following is a list of Python packages for the above code.  
```bash

```

## Usage
### cnn-example
This folder contains examples of predicting MNIST data using CNN models. `cnn-summary.md` is a simple markdown of the cnn model. To run the example:  
```bash
$ python cnn-example.py
```
or,  
```bash
$ ./cnn-example.py
```
When the code is executed, the `mnist_data` and` mnist_graph` folders are created. It stores each mnist data and learning results.

### predict-digit
The vehicle identification number is detected in the photo of the vehicle registration card and predicted. The order of execution is as follows:  
 - 

---
made by *jaejun.lee*  