# Tech-Meet-2021
**Traffic Sign Recognition**

We have worked on the high prep statement- German traffic sign recognition model along with a User Interface frame which itself allows user to augment test images and introduce new 5 classes.

This is a Deep Learning model developed using keras. It is used to recognise traffic signals. It was initially on the German Traffic Sign Recognition data set which was the baseline for this model. Later, 5 new classes were added to this model and the model was retrained to increase the difficulty of the data set. Then using an app called streamlit, the UI was created that enabled the user to play with the augmentations and visualise the results in terms of the accuracy and loss metrics. UI provided is very user friendly.

## Setup

To setup the model in your local environment please go through the following steps:

1. Make sure you have **Python 3.8** installed in your system, if not install it from [here](https://www.python.org/downloads/) and add its path to the environment variables.
>> To check if the python is correctly setup in the enviroment run `python --version` or `python3 --version`. It should print the installed version of python in your system. 
2. Run the following code to install Streamlit (It is an open source python library that we have used to integrate the model with UI): <br>
`pip install streamlit`
4. Clone this [repository](https://github.com/Reuben27/Tech-Meet-2021) in your local system.
5. Download the saved models for different types of split into validation sets from [here](https://drive.google.com/drive/folders/1UKzvVVbGQmNWUxUzGlzk0ABMG1Xbvmz6?usp=sharing) (The folder name is `saves`).
6. Move this `saves` folder in the root directory of the cloned repository in your system.
7. Download the validation datasets from here: [Test_final](https://drive.google.com/file/d/14NTQmkHsTPMXQC86hgAPFDbQ2fULLLDg/view?usp=sharing), [Test_final_2](https://drive.google.com/file/d/13dWBaCmOG_FcHH4d5GvuRRcC8iLnKYsC/view?usp=sharing). 
8. Exctract these two directories `Test_final` and `Test_final_2` in the root directory of the cloned repository in your system.
9. Open terminal and go to the location of the root directory of cloned repository in your system and run the following command: <br>
`streamlit run bosch.py`
