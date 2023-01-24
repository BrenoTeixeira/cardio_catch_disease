import numpy as np
from IPython.core.display import HTML
from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics as met
import scikitplot as skplt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import plotly.graph_objects as go
from plotly.graph_objects import Layout

def sum_table(df_):

    """This function receives a dataset and returns a dataframe with information about each column of the dataset.
    Args:
        df_ (DataFrame): Dataset.

    Returns:
        DataFrame: returns a dataframe with the number of unique values and missing value of each column of a dataset.
    """

    summary = df_.dtypes.to_frame().rename(columns={0: 'dtypes'})
    summary['Uniques'] = df_.nunique()
    summary['Missing'] = df_.isnull().sum()
    summary['Missing %'] = np.round((df_.isnull().sum()/len(df_)).values*100, 2)
    summary = summary.reset_index().rename(columns={'index': 'Name'})
    return summary

def stats_table(df):

    """Receive a dataframe and returns a dataframe with summary statistics.

    Args:
        df (DataFrame): Dataset.

    Returns:
        DataFrame: Table with summary statistics.
    """

    num_att = df.select_dtypes(include=['float', 'int'])

    return num_att.agg(['min', 'max', 'ptp', 'median', 'mean', 'std', 'var', 'skew', 'kurtosis']).T.reset_index().rename(columns={'ptp': 'range', 'index': 'attributes'})


def df_to_image(df, head_color, table_color):
    """Converts a DataFrame into a Figure

    Args:
        df (dataframe): pandas DataFrame
        head_color (str): Color of the head of the table
        table_color (str): Color of the table's cells
    """

    fig = go.Figure(data=go.Table(
        header=dict(values=list(df.columns),
                    fill_color=head_color,
                    align='center',
                    line_color='darkslategray',
                    font=dict(color='black', size=18)),
        cells=dict(values=[df[i] for i in df.columns.to_list()],
                    fill_color=table_color,
                    align='center',
                    line_color='darkslategray',
                    font=dict(color='black', size=16),
                    height=35),
        ), layout=Layout(paper_bgcolor='rgba(0,0,0,0)', width=1980, height=600))
    
    fig.show()

def note_settings():

    #%matplotlib inline
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24

    display(HTML('<style>.container {width: 100% !important; }</style>'))
    pd.options.display.max_columns = None
    pd.set_option('display.expand_frame_repr', False)

    sns.set()

def blood_pressure_label(sys, dias):

    if sys <= 120 and dias <= 80:
        return 'normal'
    elif sys < 90 and dias < 60:
        return 'low'
    elif (sys > 120 and sys <= 129) and dias < 80:
        return 'elevated'
    elif (sys >= 130 and sys <= 139) or (dias >= 80 and dias <= 89):
        return 'high_stage_1'
    elif sys >= 140 or dias >= 90:
        return 'high_stage_2'
    elif sys > 180 or dias > 120:
        'hypertensive_crisis'


def overweight_label(imc):

    if imc < 18.5  :
        return 'underweight'
    elif imc >= 18.5 and imc < 25:
        return 'healthy'
    elif imc >= 25 and imc < 30:
        return 'overweight'
    elif imc >= 30 < 35:
        return 'obesity_class1'
    elif imc >= 35 < 40:
        return 'obesity_class2'
    elif imc >= 40:
        return 'severe_obesity'


def classifier_metrics_test(clf, clf_name, x_train, x_val, y_train, y_val):


    clf.fit(x_train, y_train)
    yhat_ = clf.predict(x_val)
    y_prob = clf.predict_proba(x_val)

    precision = met.precision_score(y_val, yhat_)
    accuracy = met.accuracy_score(y_val, yhat_)
    recall = met.recall_score(y_val, yhat_)
    f1_score = met.f1_score(y_val, yhat_)
    rocauc = met.roc_auc_score(y_val, y_prob[:, 1])
    brier = met.brier_score_loss(y_val, y_prob[:, 1])

    return pd.DataFrame({clf_name: {'precision': precision, 'accuracy': accuracy, 'recall': recall, 'f1_score': f1_score, 'roc_auc': rocauc, 'brier_loss': brier}}).T


def classifier_metrics_train(clf, clf_name, x_train, y_train):


    clf.fit(x_train, y_train)
    yhat_ = clf.predict(x_train)
    y_prob = clf.predict_proba(x_train)

    precision = met.precision_score(y_train, yhat_)
    accuracy = met.accuracy_score(y_train, yhat_)
    recall = met.recall_score(y_train, yhat_)
    f1_score = met.f1_score(y_train, yhat_)
    rocauc = met.roc_auc_score(y_train, y_prob[:, 1])
    brier = met.brier_score_loss(y_train, y_prob[:, 1])

    return pd.DataFrame({clf_name+'_train': {'precision': precision, 'accuracy': accuracy, 'recall': recall, 'f1_score': f1_score, 'roc_auc': rocauc, 'brier_loss': brier}}).T


def test_set(path):

    test = pd.read_csv(path)


    ### TRANSFORMATION ###
    new_cols = ['id', 'age', 'gender', 'height', 'weight', 'systolic_pressure', 'diastolic_pressure', 'cholesterol', 'glucose', 'smoke', 'alcohol_intake', 'active', 'cardio_disease']

    test.columns = new_cols

    # Converting age from days to years
    test['age'] = test['age'].apply(lambda x: int(x/365))



    ### FEATURE ENGINEERING ###
    # Body mass Index
    test['bmi'] = test['weight']/(test['height']/100)**2

    # Weight Consition
    test['weight_condition'] = test['bmi'].apply(lambda x: overweight_label(x))

    #  Blood Pressure Level
    test['blood_pressure_level'] = test[['systolic_pressure', 'diastolic_pressure']].apply(lambda x: blood_pressure_label(x['systolic_pressure'], x['diastolic_pressure']), axis=1)

    # Cholesterol and Glucose
    levels = {1: 'normal', 2: 'above_normal', 3:'well_above_normal'}

    test['cholesterol'] = test['cholesterol'].map(levels)
    test['glucose'] = test['glucose'].map(levels)

    test['gender'] = test['gender'].apply(lambda x: 'male' if x == 2 else 'female')

    ### DATA FILTERING ###
    test = test.query('diastolic_pressure <= 140 & diastolic_pressure > 50')
    test = test.query('systolic_pressure <= 250 & systolic_pressure >= 80')
    test = test.query('systolic_pressure > diastolic_pressure')
    test = test.query('height > 65.24')

    X_test = test.drop('cardio_disease', axis=1)
    y_test = test['cardio_disease'].copy()
    
    return X_test, y_test


def classifier_metrics(clf, clf_name, x_train, x_val, y_train, y_val):

    clf.fit(x_train, y_train)
    yhat_val = clf.predict(x_val)
    y_prob_val = clf.predict_proba(x_val)

    precision_val = met.precision_score(y_val, yhat_val)
    accuracy_val = met.accuracy_score(y_val, yhat_val)
    recall_val = met.recall_score(y_val, yhat_val)
    f1_score_val = met.f1_score(y_val, yhat_val)
    rocauc_val = met.roc_auc_score(y_val, y_prob_val[:, 1])
    brier_val = met.brier_score_loss(y_val, y_prob_val[:, 1])

    validation_scores = pd.DataFrame({clf_name+'_val': {'precision': precision_val, 'accuracy': accuracy_val, 'recall': recall_val, 'f1_score': f1_score_val, 'roc_auc': rocauc_val, 'brier_loss': brier_val}}).T
    
    return  validation_scores


def classifier_metrics_tuned(clf_name, y_val, y_hat, y_prob):

    precision_train = met.precision_score(y_val, y_hat)
    accuracy_train = met.accuracy_score(y_val, y_hat)
    recall_train = met.recall_score(y_val, y_hat)
    f1_score_train = met.f1_score(y_val, y_hat)
    rocauc_train = met.roc_auc_score(y_val, y_prob[:, 1])
    brier_train = met.brier_score_loss(y_val, y_prob[:, 1])

    return pd.DataFrame({clf_name: {'precision': precision_train, 'accuracy': accuracy_train, 'recall': recall_train, 'f1_score': f1_score_train, 'roc_auc': rocauc_train, 'brier_loss': brier_train}}).T


def classifier_metrics_plot(yhat_ , y_prob, y_true, bins):

    fig, axis = plt.subplots(2, 2, figsize=(25, 15))
    
    skplt.metrics.plot_precision_recall_curve(y_true, y_prob, ax=axis[0, 0])
    CalibrationDisplay.from_predictions(y_true, y_prob[:, 1], ax=axis[0, 1], n_bins=bins)
    skplt.metrics.plot_roc(y_true, y_prob, ax=axis[1, 0])
    skplt.metrics.plot_confusion_matrix(y_true, yhat_, ax=axis[1, 1])

    plt.tight_layout()


def classifier_confusion_matrix(clf, clf_name, x_train, x_val, y_train, y_val, axis, row, col):

    clf.fit(x_train, y_train)
    yhat_ = clf.predict(x_val)
    y_prob = clf.predict_proba(x_val)

    cm = met.confusion_matrix(y_val, yhat_)

    disp = met.ConfusionMatrixDisplay(cm, display_labels=clf.classes_)

    axis[row][col].grid(False)
    axis[row][col].tick_params(axis='both', which='major', labelsize=21)
    axis[row][col].set_ylabel('True Label', fontsize=20)
    axis[row][col].set_xlabel('Predicted Label', fontsize=20)
    plt.rcParams.update({'font.size': 21})
    
    axis[row][col].set_title(f'{clf_name}', fontdict={'size': '21'})
    disp.plot(ax=axis[row][col], cmap='YlOrBr')


def classifiers_report(clf, clf_name, x_train, x_val, y_train, y_val):

    clf.fit(x_train, y_train)
    yhat_ = clf.predict(x_val)
    print(f'{clf_name}')
    print(met.classification_report(y_val, yhat_))


def scores_summary(scores_dict, classifier_name):

    dic = {}

    for metric, scores in scores_dict.items():
        mean = scores.mean()
        std = scores.std()
        ci = 1.96*std
        dic[metric] = f'{round(mean, 4)} +/- {round(ci, 4)}'
    
    return pd.DataFrame(dic, index=[classifier_name])


def calibration_metrics_comp(y_test, y_calib, y_uncalib):

    brier_calib = met.brier_score_loss(y_test, y_calib)
    brier_uncalib = met.brier_score_loss(y_test, y_uncalib)

    calib = {'brier_score_loss': brier_calib}
    uncalib = {'brier_score_loss': brier_uncalib}
    
    scores = pd.DataFrame([uncalib, calib]).T
    scores.columns = ['Uncalibrated', 'Calibrated']
    return pd.io.formats.style.Styler(scores, precision=6)    


### BUSINESS PERFORMANCE ###

diagnostic_price = lambda precision: ((precision - 0.50)//0.05) * 500
def performance(worst_precision, best_precision, num_patients, cost_per_patient):
    """_summary_

    Args:
        wort_precision (float): The lowest precision
        best_precision (float): The highest precison

    Return: A dataframe with cost, revenue and profit for each precision.
    """

    columns = ['precision', 'diagnostic_price', 'num_of_patients', 'cost', 'revenue', 'profit']

    total_cost = num_patients*cost_per_patient
    
    # Worst Case
    price_worst = diagnostic_price(worst_precision)
    revenue_worst_case = price_worst*num_patients
    profit_worst = revenue_worst_case - total_cost

    # Best Case
    price_best = diagnostic_price(best_precision)
    revenue_best_case = price_best*num_patients
    profit_best = revenue_best_case - total_cost


    performance = {'Worst Scenario': [worst_precision, price_worst, num_patients, total_cost, revenue_worst_case, profit_worst],
                   'Best Scenario': [best_precision, price_best, num_patients, total_cost, revenue_best_case, profit_best]}

    df = pd.DataFrame(performance).T
    df.columns = columns
    df[['diagnostic_price', 'cost', 'revenue', 'profit']] = df[['diagnostic_price', 'cost', 'revenue', 'profit']].applymap(lambda x: f'R${x:,.2f}')
    df['precision'] = df['precision'].apply(lambda x: f'{x:.4f}')
    
    return df


## Bootstrap ## 

def one_boot(true, pred):

    """Receives true, pred Series of the same length and the precision score.

    Args:
        true (Series): Series with the true values
        pred (Series): Series with the predict values
    """
    length = len(true)
    index = np.random.randint(0, length, size=length)

    if isinstance(true, pd.Series) and isinstance(pred, pd.Series):

        true = true.reset_index(drop=True)
        pred = pred.reset_index(drop=True)
        
        true_bt = true.values[index]
        pred_bt = pred.values[index]

    else:
        print('Type Error')

    return met.precision_score(true_bt, pred_bt)
