# Detect Box and Predict VIN-Number using Tensorflow
This repo is for Detecting box and to Predict VIN-Number based on 'Neural Network' methods. Use CNN Models and Use OpenCV package for detection.  

The project folder structure is as follows.  
```bash
$ tree -N -L 3
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
tensorflow		(1.14.0)
numpy			(1.16.3)
matplotlib		(3.2.1)
pillow			(7.1.2)

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
The vehicle identification number is detected in the photo of the vehicle registration card and predicted. Here is an introduction to python scripts:  
- `blackbox.py`: Blackbox algorithm test file that detects the vehicle identification number
- `core`
   - `common.py`: Includes mainly used functions
   - `model.py`: CNN model training file
- `main.py`: project main file
- `preprocess.py`: A file that converts a photo of a car registration certificate in the` tmp` folder into data before learning

#### Algorithm test
You can test the blackbox algorithm using `blackbox.py`. How to use is as follows:  
```bash
$ python blackbox.py sample1.jpg
contour details	 x :  1035 y :  1655 w :  132 h :  14 (1675, 1200, 3) 0.07880597014925374
...
...
 [INFO] Saving image finished
```
or,  
```bash
$ ./blackbox.py sample1.jpg
contour details	 x :  1035 y :  1655 w :  132 h :  14 (1675, 1200, 3) 0.07880597014925374
...
...
 [INFO] Saving image finished
```

In addition to sample1.jpg, the result is created in the `res/{file_name}_blackbox.jps` after execution with other jpg.  

![sample output](/blackbox_sample.jpg)

#### Preprocess Data
Preprocess the training data using `preprocess.py`. Data needed for pre-processing is moved to `tmp / box` and executed. How to use is as follows:  
```bash
$ python preprocessing.py
 [INFO] sample1.png preprocess finished!
 [INFO] sample3.png preprocess finished!
 [INFO] sample2.png preprocess finished!
```
or,  
```bash
$ ./preprocessing.py
 [INFO] sample1.png preprocess finished!
 [INFO] sample3.png preprocess finished!
 [INFO] sample2.png preprocess finished!
```

Data needed for learning is stored in a folder with the file name in the `res / char` folder.  
```bash
$ ll res/
total 1304
drwxr-xr-x   7 jj  staff     224  5 23 21:21 ./
drwxr-xr-x  12 jj  staff     384  5 23 22:20 ../
drwxr-xr-x   5 jj  staff     160  1 25 00:44 box/
drwxr-xr-x   5 jj  staff     160  5 10 15:08 char/
drwxr-xr-x   6 jj  staff     192  1 27 17:45 model_checkpoint/
-rw-r--r--   1 jj  staff  619795  5 23 21:41 sample1_blackboxed.jpg
```

#### Train Model & Prediction
**Train Model:**  
For model training, use `main.py`. Instructions to train is as follows:  
```bash
$ python main.py train {epoch: number} {batch_size: number}
```
or,  
```bash
$ ./main.py train {epoch: number} {batch_size: number}
```

**Prediction:**  
For prediction, use `main.py`. Instructions to predict is as follows:  
```bash
$ python main.py test
[PREDICT]: I [TARGET]: res/char/sample2/sample2_6.png
[PREDICT]: Q [TARGET]: res/char/sample2/sample2_7.png
[PREDICT]: P [TARGET]: res/char/sample2/sample2_5.png
[PREDICT]: F [TARGET]: res/char/sample2/sample2_4.png
[PREDICT]: H [TARGET]: res/char/sample2/sample2_0.png
[PREDICT]: N [TARGET]: res/char/sample2/sample2_1.png
[PREDICT]: S [TARGET]: res/char/sample2/sample2_3.png
[PREDICT]: M [TARGET]: res/char/sample2/sample2_2.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_15.png
[PREDICT]: 5 [TARGET]: res/char/sample2/sample2_14.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_16.png
[PREDICT]: Z [TARGET]: res/char/sample2/sample2_13.png
[PREDICT]: S [TARGET]: res/char/sample2/sample2_12.png
[PREDICT]: A [TARGET]: res/char/sample2/sample2_10.png
[PREDICT]: Z [TARGET]: res/char/sample2/sample2_11.png
[PREDICT]: R [TARGET]: res/char/sample2/sample2_9.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_8.png
```
or,  
```bash
$ ./main.py test
[PREDICT]: I [TARGET]: res/char/sample2/sample2_6.png
[PREDICT]: Q [TARGET]: res/char/sample2/sample2_7.png
[PREDICT]: P [TARGET]: res/char/sample2/sample2_5.png
[PREDICT]: F [TARGET]: res/char/sample2/sample2_4.png
[PREDICT]: H [TARGET]: res/char/sample2/sample2_0.png
[PREDICT]: N [TARGET]: res/char/sample2/sample2_1.png
[PREDICT]: S [TARGET]: res/char/sample2/sample2_3.png
[PREDICT]: M [TARGET]: res/char/sample2/sample2_2.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_15.png
[PREDICT]: 5 [TARGET]: res/char/sample2/sample2_14.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_16.png
[PREDICT]: Z [TARGET]: res/char/sample2/sample2_13.png
[PREDICT]: S [TARGET]: res/char/sample2/sample2_12.png
[PREDICT]: A [TARGET]: res/char/sample2/sample2_10.png
[PREDICT]: Z [TARGET]: res/char/sample2/sample2_11.png
[PREDICT]: R [TARGET]: res/char/sample2/sample2_9.png
[PREDICT]: B [TARGET]: res/char/sample2/sample2_8.png
```

You can check whether the prediction was successful by checking the TARGET file name.


---
made by *jaejun.lee*  
