MAKE SURE ALL OF THE REQUIRED MODULES ARE INSTALLED AND THE NLTK MODULES ARE DOWNLOADED.
NLTK MODULES ARE DOWNLOADED BY RUNNING: python popularNltk.py

PLEASE MESSAGE DESHANTAR PARAJULI ON SLACK FOR ANY ISSUES.
needs python36+

required modules: 

pip install pandas
pip install sklearn
pip install nltk

download nltk modules:

python popularNltk.py


Run the app:

python trained_model.py


Output of the app:

Do you want to train again (y/n): (TAKES A LONG TIME)
- Enter 'y' if you don't have the /local/ directory with the trained model local files
- You can hit no if the trained models are saved in the /local/ directory. If there are some issues with running the app, you may need to verify that the directory exists, finally train the model by entering 'y'.

Enter you query('exit' to end):

-End short topic based queries. Longer query means less accurate results


NB:
SVM:
KNN:
ROcchio:

-These will display the prediction from each of the classification models. SVM is the most accurate while NB is the most efficient. SVM takes a long time to train, as seen by the size of the local model, which is over 200 MB.

