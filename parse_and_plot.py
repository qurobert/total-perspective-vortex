import time
from collections import Counter

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import pywt
import warnings
import argparse
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def apply_fourier_transform(raw):
    """Apply Fourier Transform and return the frequency and amplitude."""
    try:
        data = raw.get_data()
        fft = np.fft.fft(data, axis=1)
        freq = np.fft.fftfreq(data.shape[1], d=1 / raw.info['sfreq'])

        idx = np.where((freq >= raw.info['highpass']) & (freq <= raw.info['lowpass']))
        freq = freq[idx]
        fft = fft[:, idx[0]]
    except Exception as e:
        print(f'Failed to apply Fourier Transform: {str(e)}')
        freq, fft = None, None
    return freq, np.abs(fft)


def apply_wavelet_transform(raw, wavelet='db4', level=5):
    """Apply discrete Wavelet Transform and return coefficients."""
    try:
        data = raw.get_data()
        coeffs = pywt.wavedec(data, wavelet=wavelet, level=level, axis=1)
    except Exception as e:
        print(f'Failed to apply wavelet transform: {str(e)}')
        coeffs = None
    return coeffs


def load_specific_eeg(data_dir, l_freq=8, h_freq=40, subject=None, experiment=None):
    subject_dir = os.path.join(data_dir, subject)
    edf_file = os.path.join(subject_dir, f'{subject}{experiment}.edf')

    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        filtered_raw = raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)

        freq, fft = apply_fourier_transform(filtered_raw)
        coeffs = apply_wavelet_transform(filtered_raw)

        print(f'Successfully loaded and filtered {edf_file}, with annotations')
    except Exception as e:
        print(f'Failed to load {edf_file}: {str(e)}')

    return raw, filtered_raw, freq, fft, coeffs


def parse_subject_number(subject_id):
    if subject_id <= 0 or subject_id > 109 or not isinstance(subject_id, int):
        raise ValueError('Subject ID must be between 1 and 109 inclusive.')
    if subject_id < 10:
        return f'S00{subject_id}'
    elif subject_id < 100:
        return f'S0{subject_id}'
    else:
        return f'S{subject_id}'


def plot_data_with_event(raw, start, duration, color='gray', event_color=None, show_psd=False, filtered=False):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if event_color is None:
        event_color = {id: 'r' for id in event_id.values()}

    if show_psd:
        if filtered:
            raw.plot_psd(fmin=8, fmax=40, average=True, spatial_colors=False)
        else:
            raw.plot_psd(average=True, spatial_colors=False)

    raw.plot(
        events=events,
        start=start,
        duration=duration,
        color=color,
        event_color=event_color,
        scalings='auto'
    )
    plt.show()


def parse_experiment_number(experiment_id):
    if experiment_id <= 0 or experiment_id > 14 or not isinstance(experiment_id, int):
        raise ValueError('Experiment ID must be between 1 and 14 inclusive.')
    if experiment_id < 10:
        return f'R0{experiment_id}'
    else:
        return f'R{experiment_id}'


def plot_data(raw, filtered_raw, freq, fft):
    plot_data_with_event(raw, show_psd=True, start=0, duration=10)
    plot_data_with_event(filtered_raw, show_psd=True, filtered=True, start=0, duration=10)


def extract_labels_from_annotations(raw, event_id=None):
    if event_id is None:
        event_id = {'T0': 0, 'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    labels = events[:, -1]  # Labels are in the last column
    return labels, events, event_dict


def extract_epochs(raw, events, tmin=0, tmax=1):
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=False)
    return epochs


def custom_covariance_matrix(X):
    """
    Compute the covariance matrix for each epoch.
    X: ndarray of shape (n_epochs, n_channels, n_times)
    Returns:
        cov_matrices: list of covariance matrices for each epoch
    """
    n_epochs, n_channels, n_times = X.shape
    cov_matrices = []
    for epoch in X:
        epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
        cov = (epoch @ epoch.T) / (n_times - 1)
        cov /= np.trace(cov)
        cov_matrices.append(cov)
    return cov_matrices


def custom_eigen_decomposition(matrix):
    """
    Perform eigenvalue decomposition using the power iteration method.
    matrix: symmetric matrix
    Returns:
        eigenvalues: ndarray of eigenvalues
        eigenvectors: ndarray of eigenvectors
    """
    n = matrix.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))

    residual_matrix = matrix.copy()

    for i in range(n):
        b_k = np.random.rand(n)
        for _ in range(100):
            b_k1 = residual_matrix @ b_k
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        eigenvalue = b_k.T @ residual_matrix @ b_k
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = b_k
        residual_matrix = residual_matrix - eigenvalue * np.outer(b_k, b_k)
    return eigenvalues, eigenvectors


def compute_csp(X, y, n_components=10):
    n_epochs, n_channels_times = X.shape
    n_channels = 64
    n_times = n_channels_times // n_channels
    X = X.reshape(n_epochs, n_channels, n_times)

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('CSP implementation only supports binary classification')

    class_1, class_2 = classes[0], classes[1]
    X_class0 = X[y == class_1]
    X_class1 = X[y == class_2]

    covs_class0 = custom_covariance_matrix(X_class0)
    covs_class1 = custom_covariance_matrix(X_class1)

    cov_class0 = np.mean(covs_class0, axis=0)
    cov_class1 = np.mean(covs_class1, axis=0)

    composite_cov = cov_class0 + cov_class1

    eigenvalues, eigenvectors = np.linalg.eigh(composite_cov)

    epsilon = 1e-10
    eigenvalues = np.where(eigenvalues < epsilon, epsilon, eigenvalues)

    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    S0 = whitening_matrix @ cov_class0 @ whitening_matrix.T

    eigenvalues_S0, eigenvectors_S0 = np.linalg.eigh(S0)

    sorted_indices = np.argsort(eigenvalues_S0)[::-1]
    eigenvectors_S0 = eigenvectors_S0[:, sorted_indices]

    filters = eigenvectors_S0.T @ whitening_matrix

    n_filters = n_components // 2
    selected_filters = np.vstack([filters[:n_filters], filters[-n_filters:]])

    return selected_filters


class EpochScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        n_epochs, n_features = X.shape
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled


class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        self.filters_ = compute_csp(X, y, self.n_components)
        return self

    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("You must fit the transformer before transforming data.")
        n_epochs, n_channels_times = X.shape
        n_channels = 64
        n_times = n_channels_times // n_channels
        X = X.reshape(n_epochs, n_channels, n_times)

        X_filtered = np.array([self.filters_ @ epoch for epoch in X])

        variance = np.var(X_filtered, axis=2) + 1e-10
        X_features = np.log(variance)
        return X_features


class CSPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=5):
        self.k = k
        self.selector = SelectKBest(f_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)


def train_test_val_split(X, y, test_size=0.2, val_size=0.25, random_state=42, stratify=None):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size proportionally
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def pipeline(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    class_counts = Counter(y_train)
    min_samples = min(class_counts.values())
    n_splits = min(min_samples, 10)

    print(f"Number of samples in each class: {class_counts}")
    print(f"Using {n_splits}-fold cross-validation.")

    pipeline = Pipeline([
        ('scaler', EpochScaler()),
        ('csp', CustomCSP()),
        ('classifier', LinearDiscriminantAnalysis())
    ])

    param_grid = [
        {
            'csp__n_components': [6, 8, 10, 12],
            'classifier__solver': ['svd'],
            'classifier__shrinkage': [None],
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(balanced_accuracy_score)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {grid_search.best_params_}")

    cv_scores = cross_val_score(
        grid_search.best_estimator_, X_train, y_train, cv=cv, scoring=scorer
    )

    if (X_val is None) and (y_val is None):
        print('[', end="")
        for i, score in enumerate(cv_scores):
            print(f"{score:.4f}", end="")
            if i < len(cv_scores) - 1:
                print(", ", end="")
        print(']')
    print(f"cross_validation_score: {np.mean(cv_scores):.4f}")

    if X_val is not None and y_val is not None:
        val_score = grid_search.best_estimator_.score(X_val, y_val)
        print(f"Validation set score: {val_score:.4f}")

    if X_test is not None and y_test is not None:
        test_score = grid_search.best_estimator_.score(X_test, y_test)
        print(f"Test set score: {test_score:.4f}")

    X_combined = X_train
    y_combined = y_train
    if X_val is not None and y_val is not None:
        X_combined = np.concatenate((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))

    best_model = grid_search.best_estimator_
    best_model.fit(X_combined, y_combined)

    return best_model


def simulate_data_stream(raw, pipeline, chunk_duration=1.0):
    labels, events, event_dict = extract_labels_from_annotations(raw)
    mask = labels != 0
    labels = labels[mask]
    events = events[mask]
    epochs = extract_epochs(raw, events)
    epoch_data = epochs.get_data()
    epoch_labels = epochs.events[:, -1]
    n_epochs, n_channels, n_times = epoch_data.shape

    n_features_expected = pipeline.named_steps['scaler'].scaler.n_features_in_
    n_times_expected = n_features_expected // n_channels
    chunk_size = n_times_expected

    correct_count = 0
    total_predictions = 0

    print('epoch nb: [prediction] [truth] equal?')
    for epoch_index in range(n_epochs):
        epoch = epoch_data[epoch_index]
        true_label = epoch_labels[epoch_index]
        n_samples = epoch.shape[1]

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = epoch[:, start:end]

            if chunk.shape[1] < chunk_size:
                pad_width = chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='edge')

            chunk_data = chunk.reshape(1, -1)

            start_time = time.time()
            prediction = pipeline.predict(chunk_data)
            end_time = time.time()

            delay = end_time - start_time

            if delay > 2.0:
                print(f"Warning: Processing time exceeded 2 seconds for chunk starting at {start / sfreq:.2f}s")

            if prediction[0] == true_label:
                correct_count += 1

            total_predictions += 1
            if epoch_index < 10:
                epoch_index = f"0{epoch_index}"
            if prediction[0] == true_label:
                equal = True
            else:
                equal = False
            print(f"epoch {epoch_index}: \t"
                  f"[{prediction[0]}]\t[{true_label}] {equal}\tDelay {delay:.4f}s")

    accuracy = correct_count / total_predictions if total_predictions > 0 else 0
    print(f"Stream Accuracy: {accuracy:.4f}")


def get_epoch_data_from_raw(raw):
    labels, events, event_dict = extract_labels_from_annotations(raw)
    mask = labels != 0
    labels = labels[mask]
    events = events[mask]
    epochs = extract_epochs(raw, events)
    if len(epochs) == 0:
        raise ValueError("No epochs extracted.")
    epoch_data = epochs.get_data()
    epoch_labels = epochs.events[:, -1]
    n_epochs, n_channels, n_times = epoch_data.shape
    epoch_data = epoch_data.reshape((n_epochs, n_channels * n_times))
    return epoch_data, epoch_labels


def training_all_subjects(directory, experiments=None, printAccuracy=False):
    preprocessed_data_file = 'preprocessed_data.pkl'

    if os.path.exists(preprocessed_data_file):
        print('Loading preprocessed data...')
        data = joblib.load(preprocessed_data_file)
        X_train_all = data['X_train_all']
        y_train_all = data['y_train_all']
        X_val_all = data['X_val_all']
        y_val_all = data['y_val_all']
        test_data = data['test_data']
    else:
        if experiments is None:
            experiments = range(3, 14)
        subjects = list(range(1, 110))
        X_train_all, y_train_all = [], []
        X_val_all, y_val_all = [], []
        test_data = {}

        max_dim = 0

        for subject_id in subjects:
            subject = parse_subject_number(subject_id)
            for experiment_id in experiments:
                experiment = parse_experiment_number(experiment_id)
                try:
                    raw, filtered_raw, _, _, _ = load_specific_eeg(
                        directory, subject=subject, experiment=experiment
                    )
                    epoch_data, _ = get_epoch_data_from_raw(filtered_raw)
                    n_epochs, n_features = epoch_data.shape
                    max_dim = max(max_dim, n_features)
                except Exception as e:
                    print(f'Error processing subject {subject}, experiment {experiment}: {e}')

        for subject_id in subjects:
            subject = parse_subject_number(subject_id)
            for experiment_id in experiments:
                experiment = parse_experiment_number(experiment_id)
                try:
                    raw, filtered_raw, _, _, _ = load_specific_eeg(
                        directory, subject=subject, experiment=experiment
                    )
                    epoch_data, epoch_labels = get_epoch_data_from_raw(filtered_raw)

                    if epoch_data.shape[1] < max_dim:
                        pad_width = ((0, 0), (0, max_dim - epoch_data.shape[1]))
                        epoch_data = np.pad(epoch_data, pad_width, mode='constant', constant_values=0)

                    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
                        epoch_data, epoch_labels, stratify=epoch_labels
                    )

                    X_train_all.append(X_train)
                    y_train_all.append(y_train)
                    X_val_all.append(X_val)
                    y_val_all.append(y_val)

                    if experiment_id not in test_data:
                        test_data[experiment_id] = {'X_test': [], 'y_test': []}
                    test_data[experiment_id]['X_test'].append(X_test)
                    test_data[experiment_id]['y_test'].append(y_test)
                except Exception as e:
                    print(f'Error processing subject {subject}, experiment {experiment}: {e}')

        X_train_all = np.concatenate(X_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all)
        X_val_all = np.concatenate(X_val_all, axis=0)
        y_val_all = np.concatenate(y_val_all)
        print('Saving preprocessed data...')
        data = {
            'X_train_all': X_train_all,
            'y_train_all': y_train_all,
            'X_val_all': X_val_all,
            'y_val_all': y_val_all,
            'test_data': test_data
        }
        joblib.dump(data, preprocessed_data_file)
        print('Preprocessed data saved.')

    print('Starting training on all subjects and experiments...')

    model = pipeline(X_train_all, y_train_all, X_val=X_val_all, y_val=y_val_all)
    print('Training completed.')

    if printAccuracy:
        print_accuracy(model, test_data, experiments)
    return model


def print_accuracy(model, test_data, experiments):
    overall_mean_accuracies = []
    print('Mean accuracy of the six different experiments for all 109 subjects:')
    for i, experiment_id in enumerate(experiments):
        X_test_all = np.concatenate(test_data[experiment_id]['X_test'], axis=0)
        y_test_all = np.concatenate(test_data[experiment_id]['y_test'], axis=0)
        test_score = model.score(X_test_all, y_test_all)
        overall_mean_accuracies.append(test_score)
        print(f'experiment {i}:\taccuracy = {test_score:.4f}')

    overall_mean_accuracy = np.mean(overall_mean_accuracies)
    print(f'Mean accuracy of 6 experiments: {overall_mean_accuracy:.4f}')


def main():
    directory = './datasets'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('subject_id', nargs='?', type=int, help='ID of the subject')
    parser.add_argument('experiment_id', nargs='?', type=int, help='ID of the experiment')
    parser.add_argument('mode', nargs='?', choices=['train', 'predict'],
                        help='Mode of operation: train or predict')
    args = parser.parse_args()

    if not args.mode or not args.subject_id or not args.experiment_id:
        print('Training model on all subjects and experiments.')
        experiments = [3, 4, 7, 8, 11, 12]
        training_all_subjects(directory, experiments, printAccuracy=True)
        return 0

    try:
        subject = parse_subject_number(args.subject_id)
        experiment = parse_experiment_number(args.experiment_id)
    except ValueError as e:
        print(e)
        return 1


    raw, filtered_raw, freq, fft, coeffs = load_specific_eeg(directory, subject=subject,
                                                             experiment=experiment)
    plot_data(raw, filtered_raw, freq, fft)

    epoch_data, epoch_labels = get_epoch_data_from_raw(filtered_raw)
    print(f"Total number of epochs: {len(epoch_labels)}")

    if args.mode == 'train':
        print(f'Training model on subject {args.subject_id} and experiment {args.experiment_id}.')
        pipeline(epoch_data, epoch_labels)
    elif args.mode == 'predict':
        print(f'Predicting on subject {args.subject_id} and experiment {args.experiment_id}')
        if os.path.exists('model.pkl'):
            model = joblib.load('model.pkl')
            print("Model loaded.")
        else:
            model = training_all_subjects(directory, None, False)
            joblib.dump(model, 'model.pkl')
            print("Model saved.")
        simulate_data_stream(filtered_raw, model)
    else:
        print('Invalid mode of operation. Please choose either "train" or "predict".')


if __name__ == '__main__':
    main()