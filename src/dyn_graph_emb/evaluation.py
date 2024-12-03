from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pprint
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score, roc_curve, auc, precision_score, accuracy_score,
    confusion_matrix, recall_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def calculate_stat_dicts(dict_list):
    combined_values = {}
    for d in dict_list:
        for key, value in d.items():
            if key in combined_values:
                combined_values[key].append(value)
            else:
                combined_values[key] = [value]

    result = {key: {'mean': np.mean(values), 'sd': np.std(values, ddof=0)} for key, values in combined_values.items()}

    return result


def pretty_print_dict(d):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(d)
    print()


def train_multiclass_v2(xs, ys, n_splits=5, to_print=True, random_state=0, kernel='linear'):
    is_binary = (len(ys.shape) == 1) or (ys.shape[1] == 1)
    if not is_binary:
        int_labels = np.argmax(ys, axis=1)
    else:
        ys = ys if (len(ys.shape) == 1) else ys.flatten()
        int_labels = ys
    n_splits = int(np.min([n_splits, np.min(np.sum(ys, axis=0))]))
    print(f'n splits: {n_splits}')
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(xs, int_labels)
    splits = kf.split(xs, int_labels)
    results = []

    for i, (train_idx, test_idx) in tqdm(enumerate(splits), total=n_splits):
        np.random.seed(random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(xs[train_idx])
        X_test_scaled = scaler.transform(xs[test_idx])
        clf = SVC(kernel=kernel, probability=True, random_state=random_state)
        if not is_binary:
            clf.fit(X_train_scaled, np.argmax(ys[train_idx], axis=1))
        else:
            clf.fit(X_train_scaled, ys[train_idx])
        rat = np.sum(ys[test_idx])/len(ys[test_idx])
        print(f'ratio: {rat}, {1-rat}')
        result = evaluate_classifier(X_test_scaled, ys[test_idx], clf)
        print()
        results.append(result)

    stat_result = calculate_stat_dicts(results)
    if to_print:
        pretty_print_dict(stat_result)

    return stat_result


def train_multiclass(xs, ys, n_splits=5, to_print=True, random_state=0):
    is_binary = (len(ys.shape) == 1) or (ys.shape[1] == 1)
    if not is_binary:
        int_labels = np.argmax(ys, axis=1)
    else:
        ys = ys if (len(ys.shape) == 1) else ys.flatten()
        int_labels = ys
    n_splits = int(np.min([n_splits, np.min(np.sum(ys, axis=0))]))
    print(f'n splits: {n_splits}')
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(xs, int_labels)
    splits = kf.split(xs, int_labels)
    results = []

    for i, (train_idx, test_idx) in tqdm(enumerate(splits), total=n_splits):
        np.random.seed(random_state)
        clf = LogisticRegression(max_iter=500, random_state=0)
        if not is_binary:
            clf.fit(xs[train_idx], np.argmax(ys[train_idx], axis=1))
        else:
            clf.fit(xs[train_idx], ys[train_idx])
        result = evaluate_classifier(xs[test_idx], ys[test_idx], clf)
        results.append(result)

    stat_result = calculate_stat_dicts(results)
    if to_print:
        pretty_print_dict(stat_result)

    return stat_result


def evaluate_classifier(X, Y, classifier):
    y_pred = classifier.predict(X)
    y_scores = classifier.predict_proba(X)
    is_binary = False
    if len(Y.shape) == 1 or Y.shape[1] == 1:
        is_binary = True

    if not is_binary:
        # Convert one-hot encoded Y back to class labels for some metrics
        y_true = np.argmax(Y, axis=1)
    else:
        y_true = Y.flatten() if len(Y.shape) > 1 else Y
        y_scores = y_scores[:, 1][:, np.newaxis]

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    average_precision_macro = average_precision_score(Y, y_scores, average='macro')
    average_precision_micro = average_precision_score(Y, y_scores, average='micro')

    auc_macro = roc_auc_score(Y, y_scores, average='macro')
    auc_micro = roc_auc_score(Y, y_scores, average='micro')

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate sensitivity (recall)
    sensitivity = recall_score(y_true, y_pred)
    print(f"Sensitivity: {sensitivity:.2f}")

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.2f}")

    return {
        'Average Precision Score (Macro)': average_precision_macro,
        'Average Precision Score (Micro)': average_precision_micro,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Micro)': f1_micro,
        'AUC (Macro)': auc_macro,
        'AUC (Micro)': auc_micro,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
    }
