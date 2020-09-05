# Detect Box and Predict VIN-Number using Tensorflow
This repo is for Detecting box and to Predict VIN-Number based on 'Neural Network' methods. Use CNN Models and Use OpenCV package for detection.  

## How to use
### install packages  
```bash
$ pip install -r requirements.txt
```

### Blackbox Tutorial
```bash
$ python main.py
[Detect VIN Num Using BlackBox Algorithm]
Usage:
******************************************
****** 1) CNN Tutorial                  **
****** 2) Blackbox Tutorial             **
****** 3) Train model                   **
****** 4) Test model                    **
******************************************
Select Number:
```

### CNN Tutorial
```bash
$ python main.py
[Detect VIN Num Using BlackBox Algorithm]
Usage:
******************************************
****** 1) CNN Tutorial                  **
****** 2) Blackbox Tutorial             **
****** 3) Train model                   **
****** 4) Test model                    **
******************************************
Select Number: 1
```  
When the code is executed, the `mnist_data`, `mnist_graph`, `checkpoints` folders are created. It stores each mnist
data and learning results.

### Blackbox Tutorial
You can test the blackbox algorithm . How to use is as follows:  
```bash
$ python main.py
[Detect VIN Num Using BlackBox Algorithm]
Usage:
******************************************
****** 1) CNN Tutorial                  **
****** 2) Blackbox Tutorial             **
****** 3) Train model                   **
****** 4) Test model                    **
******************************************
Select Number: 2
```  
In addition to sample1.jpg, the result is created in the `res/{file_name}_blackbox.jps` after execution with other jpg.  
![sample output](img/blackbox_sample.jpg)

### Train Model & Prediction
**Train Model:**  
For model training, Enter Num `3`. Instructions to train is as follows:  
```bash
[Detect VIN Num Using BlackBox Algorithm]
Usage:
******************************************
****** 1) CNN Tutorial                  **
****** 2) Blackbox Tutorial             **
****** 3) Train model                   **
****** 4) Test model                    **
******************************************
Select Number: 3
...
```

**Prediction:**  
For prediction, Enter Num `4`. Instructions to train is as follows:  
```bash
[Detect VIN Num Using BlackBox Algorithm]
Usage:
******************************************
****** 1) CNN Tutorial                  **
****** 2) Blackbox Tutorial             **
****** 3) Train model                   **
****** 4) Test model                    **
******************************************
Select Number: 4
```  

You can check whether the prediction was successful by checking the TARGET file name. `(res/char/{sample_image_name})`

---

made by *jaejun.lee*  