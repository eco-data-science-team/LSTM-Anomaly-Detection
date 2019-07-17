# Anomaly Detection
 LSTM (Long Short Term Memory) Network

## Intsall Requirements to use this repo
In Terminal:

        $ pip install --user --requirements requirements.txt

## Folder Structure
1. **Model** - Folder that contains saved LSTM model architecture and weights. Used in `predictor.py`
2. **config** - Folder that contains:
   
    - `lstmconfig.ini` : used by the `lstmGrid.py`, `singleLSTM.py` and `LSTM Notebook.ipynb` to load variables
    - `predictorconfig.ini` : used by `predictor.py` to load variables such as the name of the model's architecture and it's respective weights
3. `LSTM Notebook.ipynb` : Jupyter Notebook used to debug and view the predicitions of a single LSTM model
4. `lstmGrid.py` : script that implements a grid search using  scikit-learn's `GridSearchCV`. **Generates** three files: a pickled object of all the grid search results, the architecture and weights of the best model.
5. `predictor.py` : script used to run the model created by the `lstmGrid.py` script.
6. `singleLSTM.py` : script that iterates over a user specified (**n_jobs** in the `lstmconfig.ini` file) amount of runs and saves the run's epochs vs loss metrics into a plot. **Generates** two additional files from the plot which are, the last run's model architecture along with its weights.

### Simple Folder Structure

![FOLDER_STRUCT](LSTM_Folder_Struct.png)