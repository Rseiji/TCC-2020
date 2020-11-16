import pandas as pd
import seaborn as sns
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def compile_results(flat_predictions, flat_true_labels, test_set):
    """Joins test set and predicted label's results.
    """
    result_concatenation = pd.DataFrame({'pred': flat_predictions, 'labels_tmp': flat_true_labels})
    compiled_results = pd.concat([test_set, result_concatenation], axis=1)

    assert (compiled_results['label'] == compiled_results['labels_tmp']).all()
    compiled_results = compiled_results.drop(columns=['labels_tmp'])
    return compiled_results


def get_test_set_analysis(df):
    labels_0 = df[df['label'] == 0]['label']
    preds_0 = df[df['label'] == 0]['pred']

    labels_1 = df[df['label'] == 1]['label']
    preds_1 = df[df['label'] == 1]['pred']

    labels = df['label']
    preds = df['pred']

    results = pd.DataFrame()

    results.loc['accuracy', 'label_0'] = accuracy_score(labels_0, preds_0)
    results.loc['accuracy', 'label_1'] = accuracy_score(labels_1, preds_1)
    results.loc['accuracy', 'complete_dataset'] = accuracy_score(labels, preds)

    results.loc['balanced_acc', 'label_0'] = balanced_accuracy_score(labels_0, preds_0)
    results.loc['balanced_acc', 'label_1'] = balanced_accuracy_score(labels_1, preds_1)
    results.loc['balanced_acc', 'complete_dataset'] = balanced_accuracy_score(labels, preds)

    results.loc['precision_score', 'label_0'] = precision_score(labels_0, preds_0)
    results.loc['precision_score', 'label_1'] = precision_score(labels_1, preds_1)
    results.loc['precision_score', 'complete_dataset'] = precision_score(labels, preds)

    results.loc['recall', 'label_0'] = recall_score(labels_0, preds_0)
    results.loc['recall', 'label_1'] = recall_score(labels_1, preds_1)
    results.loc['recall', 'complete_dataset'] = recall_score(labels, preds)

    results.loc['f1_score', 'label_0'] = f1_score(labels_0, preds_0)
    results.loc['f1_score', 'label_1'] = f1_score(labels_1, preds_1)
    results.loc['f1_score', 'complete_dataset'] = f1_score(labels, preds)

    conf_mat = confusion_matrix(df['label'], df['pred'])
    return (results, conf_mat)


def save_analysis(results, conf_mat, file_name_analysis, file_name_conf_mat, path=''):
    results.to_csv(path + file_name_analysis)
    sns.heatmap(conf_mat, annot=True).get_figure().savefig(file_name_conf_mat)


def save_model(model, model_pickle_name):
    with open(model_pickle_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_results(results, results_pickle_name):
    with open(results_pickle_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

