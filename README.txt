- Installation of Packages -
pip install numpy 
pip install pandas
pip install tensorflow
pip install scipy
pip install matplotlib
pip install seaborn
pip install sklearn

- Requirements -
numpy == 1.20.1
pandas == 1.2.4
tensorflow == 2.7.0
scipy == 1.6.2
matplotlib == 3.3.4
seaborn == 0.11.1
sklearn == 1.1.3


- Usage -
To run subject specific model results for Bio-Z to BP estimation run:
	python subject_specific_regression.py

To run the initial leave one subject out analysis for Bio-Z to BP estimation run:
	python one_subject_out_regression.py

To run the supplementary leave subject out analysis for Bio-Z to BP estimation run:
	python leave_subjects_out_regression.py