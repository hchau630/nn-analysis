### REQUIRES SCIKIT-LEARN >= 24.0 FOR RIDGECV ALPHA_PER_TARGET=TRUE ###
import warnings

from sklearn.linear_model import Ridge, RidgeCV, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from scipy.stats import spearmanr
import torch
import numpy as np

from nn_analysis import utils

@utils.numpy_to_torch
def cc(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        raise NotImplementedError()
    v1, v2 = y - y.mean(dim=0), pred_y - pred_y.mean(dim=0)
    return (v1*v2).sum(dim=0)/(v1.norm(dim=0)*v2.norm(dim=0))

@utils.numpy_to_torch
def r2(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        raise NotImplementedError()
    return 1 - ((y - pred_y).norm(dim=0)**2)/((y - y.mean(dim=0)).norm(dim=0)**2)

@utils.numpy_to_torch
def mse(pred_y, y, weights=None):
    # pred_y, y - (n_samples, n_targets) or (n_samples)
    # return (n_targets)
    if weights is not None:
        return (weights*(y - pred_y)**2).mean(dim=0)
    return ((y - pred_y)**2).mean(dim=0)

def cosine_similarity(v1, v2):
    # v1, v2 - (n_targets, n_features) or (n_features)
    # returns (n_targets)
    if len(v1.shape) == 1:
        v1 = np.expand_dims(v1, 0)
    if len(v2.shape) == 1:
        v2 = np.expand_dims(v2, 0)
    v1_lens = np.sqrt(np.sum(v1**2, axis=1))
    v2_lens = np.sqrt(np.sum(v2**2, axis=1))
    s = np.sum(v1*v2, axis=1)/(v1_lens*v2_lens)
    return s

def compute_neural_fits(X, y, clf=RidgeCV(alphas=np.logspace(-2,6,9)), n_splits=5, test_size=0.2):
    """
    X, y - (images, neural net neurons), (images, real neurons, trials)
    """
    corrected_ccs, alphas = np.zeros(n_splits), np.zeros((n_splits, 2, y.shape[1])) # (n_splits), (n_splits, 2, real neurons)
    
    y = np.transpose(y, axes=(2,0,1)) # (trials, images, real neurons)
    
    for split in range(n_splits):
        # Randomly split data into two halves, with random_state specified so that the splitting is consistent
        # across different models and different neural data
        y_0, y_1 = train_test_split(y, test_size=0.5, random_state=split) # Splits along axis 0 (trials)
        y_0, y_1 = np.mean(y_0, axis=0), np.mean(y_1, axis=0) # Get mean firing rates (images, real neurons)

        # Split into training and test sets for computing cc. Specify random_state for consistent splits
        train_X, test_X, train_y_0, test_y_0, train_y_1, test_y_1 = train_test_split(
            X, y_0, y_1, test_size=test_size, random_state=split
        ) # (train_images, fake neurons), (test_images, fake neurons), (train_images, real neurons), (test_images, real neurons), ...

        clf.fit(train_X, train_y_0) # (train_images, fake neurons), (train_images, real neurons)
        pred_y_0 = clf.predict(test_X) # (test_images, real neurons)
        cc_0 = cc(pred_y_0, test_y_0) # (real neurons)
        alphas[split,0] = clf.alpha_ # (real neurons)

        clf.fit(train_X, train_y_1) # (train_images, fake neurons), (train_images, real neurons)
        pred_y_1 = clf.predict(test_X) # (test_images, real neurons)
        cc_1 = cc(pred_y_1, test_y_1) # (real neurons)
        alphas[split,1] = clf.alpha_ # (real neurons)

        cc_model_data = (cc_0 + cc_1)/2 # (real neurons)
        cc_model_model = cc(pred_y_0, pred_y_1) # (real neurons)
        cc_data_data = cc(test_y_0, test_y_1) # (real neurons)

#         print("NaNs: ", torch.isnan(cc_model_data / torch.sqrt(cc_model_model*cc_data_data)).sum())
        corrected_cc = np.nanmean((cc_model_data / torch.sqrt(cc_model_model*cc_data_data)).numpy()) # (1)
        corrected_ccs[split] = corrected_cc

    return np.mean(corrected_ccs), np.mean(alphas, axis=(0,1)) # (1), (real neurons)

def compute_classification_scores(X, y, cv_regs=np.logspace(-4,4,9), cv=5, test_size=0.2, random_state=0, clf_type='svc', stratify=True, suppress=False):
    if stratify:
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if clf_type == 'log_reg':
        raise NotImplementedError() # NOT SURE, NEED CHECKING BEFORE USING THIS
        clf = LogisticRegressionCV(Cs=cv_regs, cv=cv, n_jobs=-1)
        clf.fit(train_x, train_y)
        return clf.score(test_X, test_y), clf.C_
    elif clf_type == 'linear_svc':
        svc = LinearSVC()
    elif clf_type == 'svc':
        svc = SVC(kernel='linear')
    else:
        raise NotImplementedError("clf_type must be one of 'log_reg', 'linear_svc', or 'svc'")
    parameters = {'C': cv_regs}
    clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=-1) # Parallelize
    if suppress:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train_X, train_y)
    else:
        clf.fit(train_X, train_y)
    svc = clf.best_estimator_
    
    return svc.score(test_X, test_y), clf.best_params_['C']

def compute_regression_scores(X, y, cv_regs=np.logspace(-2,6,9), cv=5, test_size=0.2, random_state=0, suppress=False, **kwargs):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf = RidgeCV(
        alphas=cv_regs,
        alpha_per_target=False,
        cv=cv,
    )
    if suppress:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train_X, train_y)
    else:
        clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    return cc(pred_y, test_y), clf.alpha_

def compute_single_ccg_score(X_1, y_1, X_2, y_2, cv_reg, suppress=False):
    """
    X_1, X_2 - (n_samples, n_features)
    y_1, y_2 - (n_samples)
    """ 
    assert len(y_1.shape) == 1 and len(y_2.shape) == 1 # Only decode 1 variable
    clf_1 = Ridge(alpha=cv_reg)
    clf_2 = Ridge(alpha=cv_reg)
    
    if suppress:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_1.fit(X_1, y_1) # Fit clf_1 on first set of data
            clf_2.fit(X_2, y_2) # Fit clf_2 on second set of data
    else:
        clf_1.fit(X_1, y_1) # Fit clf_1 on first set of data
        clf_2.fit(X_2, y_2) # Fit clf_2 on second set of data
    
    pred_y_1, pred_y_2 = clf_2.predict(X_1), clf_1.predict(X_2)
    
    ccg_cc_1 = cc(pred_y_2, y_2) # (1) test clf_1 performance on second set of data
    ccg_cc_2 = cc(pred_y_1, y_1) # (1) test clf_2 performance on first set of data
    ccg_cc = 0.5*(ccg_cc_1 + ccg_cc_2)
    
    ccg_r2_1 = r2(pred_y_2, y_2) # (1) test clf_1 performance on second set of data
    ccg_r2_2 = r2(pred_y_1, y_1) # (1) test clf_2 performance on first set of data
    ccg_r2 = 0.5*(ccg_r2_1 + ccg_r2_2)
    
    parallelism = cosine_similarity(clf_1.coef_, clf_2.coef_) # (1) (train_clf.coef_ has shape (n_features))
    
    return {'ccg_cc': ccg_cc, 'ccg_r2': ccg_r2, 'parallelism': parallelism}

def compute_ccg_scores(X, y, metrics, train_size=0.125, test_size=0.25, cv_regs=np.logspace(-2,6,9), log=False, **kwargs):
    """
    X - (n_base_images, n_manips, n_features)
    y - (n_base_images, n_manips)
    """
    assert len(X.shape) == 3 and len(y.shape) == 2

    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    n_train, n_test = train_X.shape[0], test_X.shape[0]
    
    count = 0
    total = len(cv_regs)*n_train*(n_train-1)//2
    train_results = {metric: np.zeros((len(cv_regs),n_train,n_train)) for metric in metrics}
    for k, cv_reg in enumerate(cv_regs):
        for i in range(n_train):
            for j in range(i+1, n_train):
                train_result = compute_single_ccg_score(train_X[i], train_y[i], train_X[j], train_y[j], cv_reg, **kwargs)
                for metric in metrics:
                    train_results[metric][k,i,j] = train_result[metric]
                count += 1
                if log and count % (total // 10) == 0:
                    print(f'Train progress: {count}/{total}')
    
    indices = np.triu_indices(n_train,k=1)
    avg_train_results = {metric: np.zeros(len(cv_regs)) for metric in metrics}
    for i in range(len(cv_regs)):
        for metric in metrics:
            avg_train_results[metric][i] = train_results[metric][i][indices].mean()
    
    reg_params = {metric: cv_regs[avg_train_results[metric].argmax()] for metric in metrics}
    
    count = 0
    total = n_test*(n_test-1)//2
    results = {metric: np.zeros((n_test,n_test)) for metric in metrics}
    for i in range(n_test):
        for j in range(i+1, n_test):
            for metric in metrics:
                results[metric][i,j] = compute_single_ccg_score(test_X[i], test_y[i], test_X[j], test_y[j], reg_params[metric], **kwargs)[metric]
            count += 1
            if log and count % (total // 5) == 0:
                print(f'Test progress: {count}/{total}')
    
    indices = np.triu_indices(n_test,k=1)
    results = {metric: results[metric][indices].mean() for metric in metrics}
    
    return results, reg_params

def compute_decoding_metric(X, y, method, **kwargs):
    """
    X - (n_samples, n_features)
    y - (n_samples)
    """
    if method == 'classification':
        return compute_classification_scores(X, y, **kwargs)
    elif method == 'regression':
        return compute_regression_scores(X, y, **kwargs)
    else:
        raise NotImplementedError("method should be either 'classification' or 'regression'.")

def compute_factorization_metrics(acts, subsample_units=None, threshold=0.9):
    #acts should have shape N x K x D, where
    #D = number of units/neurons
    #K is the number of manipulations per "base image" (e.g. number of different crops)
    #N is the number of "base images"
    #subsample_units (optional) -- subsample a fixed number of units to ensure standardization across models
    #threshold -- variance threshold for factorization metrics (see below)
    #The code will compute the "factorization" of the variance due to the "manipulations" (and other quantities, see below)
    #This can be interpreted loosely: for instance: "manipulations" could mean varying object position, background, etc.
    #Three quantities will be returned as a tuple: (factorization, disentanglement, invariance)
    #Here are my definitions for these terms (not standard terminology):
    #Invariance = "What fraction of the variance is not manipulation-related" (so 1.0 is perfect invariance)
    #Disentanglement = "What fraction of the manipulation-induced variance resides outside of the non-manipulation-related subspace" (1.0 is perfect disentanglement)
    #Factorization = "In the non-manipulation-related subspace, what fraction of the variance is not manipulation-related" (1.0 is perfect factorization)
    #The "non-manipulation-related subspace" is defined as the top M pcs of the non-manipulation-related variance, where M is the minimum required to explain {threshold*100}% of the variance
    
    if subsample_units is not None:
        acts = acts[:, :, np.random.permutation(acts.shape[2])[:subsample_units]]

    acts_manip_mean = np.mean(acts, 1)
    acts_flat = np.reshape(acts, [-1, acts.shape[2]])
    acts_centered = acts - np.mean(acts, 1, keepdims=True)
    acts_centered = np.reshape(acts_centered, [-1, acts_centered.shape[2]])
    var_from_manip = np.var(acts_centered, axis=0).sum()
    var_other = np.var(acts_manip_mean, axis=0).sum()

    pca = PCA(n_components=threshold)
    pca.fit(acts_manip_mean) # compute PCs due to non-manipulation-related variance

    var_from_manip_proj = np.var(pca.transform(acts_centered), axis=0).sum() #within-image variance in cross-image dim


    totvar_proj = np.var(pca.transform(acts_flat), axis=0).sum()
    var_total = np.var(acts_flat, axis=0).sum()

    if threshold == 1.0:
        var_from_manip_proj = np.var(acts_centered, axis=0).sum()
        totvar_proj = np.var(acts_flat, axis=0).sum()
        
        
    results = {
        "ss_inv": 1 - (var_from_manip_proj / totvar_proj), 
        "fact": 1 - (var_from_manip_proj / var_from_manip), 
        "inv": var_other / var_total
    }
    return results

def compute_dimensionality(cumulative_evr):
    assert len(cumulative_evr.shape) == 1
    evr = cumulative_evr[1:] - cumulative_evr[:-1]
    participation_ratio = np.sum(evr)**2/np.sum(evr**2)
    score = (participation_ratio-1)/(len(evr)-1)
    return score

def compute_sparsity(y, zero_one=True):
    n_images, n_neurons = y.shape[0], y.shape[1]
    if zero_one:
        y = y - np.min(y, axis=0, keepdims=True)
        y = y / (1.0e-8+np.max(y, axis=0, keepdims=True))
    population_sparsity = np.mean((n_neurons - np.sum(y, axis=-1)**2/np.sum(y**2, axis=-1))/(n_neurons-1))
    trial_sparsity = np.mean((n_images - np.sum(y, axis=0)**2/(1.0e-8+np.sum(y**2, axis=0)))/(n_images-1))
    return {'population': population_sparsity, 'trial': trial_sparsity}

def compute_rdm(X, metric='pearson'):
    """
    X - (images, neurons) or (images, neurons, trials)
    """
    if X.ndim == 2:
        if metric == 'pearson':
            correlations = 1.0 - np.corrcoef(X)
        elif metric == 'spearman':
            correlations = 1.0 - spearmanr(X, axis=1)[0]
        else:
            raise RuntimeError("metric must be 'pearson' or 'spearman'")
        assert correlations.shape[0] == X.shape[0]
        return correlations # (images, images)
    elif X.ndim == 3:
        rdms = [compute_rdm(X[...,i]) for i in range(X.shape[-1])]
        return np.stack(rdms,axis=0) # (trials, images, images)
    else:
        raise RuntimeError("X must have shape (images, neurons) or (images, neurons, trials)")

def compute_rdm_neural_fits(model_rdm, neural_rdms, n_splits=3, metric='spearman', random_state=False):
    assert model_rdm.shape == neural_rdms.shape[1:]
    assert model_rdm.ndim == 2
    assert model_rdm.shape[0] == model_rdm.shape[1]

    indices = np.triu_indices(model_rdm.shape[0], k=1)
    model_vec, neural_vecs = model_rdm[indices], neural_rdms[:,indices[0],indices[1]]
    assert len(model_vec) == (model_rdm.shape[0]-1)*model_rdm.shape[0]//2
    
    if metric == 'pearson':
        func = lambda vec_1, vec_2: np.corrcoef(vec_1, vec_2)[0,1]
    elif metric == 'spearman':
        func = lambda vec_1, vec_2: spearmanr(vec_1, vec_2)[0]
    else:
        raise RuntimeError("metric must be 'pearson' or 'spearman'")
    
    scores = np.zeros(n_splits)
    for split in range(n_splits):
        state = None if random_state else split
        neural_vecs_0, neural_vecs_1 = train_test_split(neural_vecs, test_size=0.5, random_state=state)
        neural_vec_0, neural_vec_1 = np.mean(neural_vecs_0, axis=0), np.mean(neural_vecs_1, axis=0)
        
        score_0 = func(model_vec, neural_vec_0)
        score_1 = func(model_vec, neural_vec_1)
        
        score_model_data = (score_0 + score_1) / 2
        score_data_data = func(neural_vec_0, neural_vec_1)
        
        scores[split] = score_model_data / score_data_data**0.5
    return np.mean(scores)

def compute_curvature(X):
    """
    Computes the angle between the displacement vectors at successive frames, normalized
    between 0 and 1.
    Input:
    X - (n_frames, n_features)
    
    Returns:
    scores - (n_frames - 2)
    """
    vecs = X[1:] - X[:-1] # displacement vectors
    vecs = vecs/np.linalg.norm(vecs, axis=-1, keepdims=True) # normalize the displacement vectors
    dots = np.einsum('ni,ni->n',vecs[1:],vecs[:-1]) # dot product between successive normalized displacement vectors
    angles = np.arccos(dots) # angles in radians
    angles = angles/np.pi # normalize to 1, since np.arccos outputs values between 0 and pi.
    return angles
