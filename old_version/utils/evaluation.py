from .identifier_splitting import split_identifier_into_parts
import numpy as np

def calculate_precision(prediction, ground_truth):
    ground_truth_splits = split_identifier_into_parts(ground_truth)
    prediction_splits = split_identifier_into_parts(prediction)
    correct_results = 0
    all_returned_results = 0 
    for split in prediction_splits:
        if split in ground_truth_splits:
            correct_results += 1
        all_returned_results += 1
    precision = float(correct_results/all_returned_results)
    return precision
    
def calculate_recall(prediction, ground_truth): 
    ground_truth_splits = split_identifier_into_parts(ground_truth)
    should_have_returned_results = len(ground_truth_splits)
    prediction_splits = split_identifier_into_parts(prediction)
    correct_results = 0
    for split in prediction_splits:
        if split in ground_truth_splits:
            correct_results += 1
    
    recall = float(correct_results/should_have_returned_results)
    return recall
    
def calculate_f1_score(prediction, ground_truth):
    precision = calculate_precision(prediction, ground_truth)
    recall = calculate_recall(prediction, ground_truth)
    f1_score = 0
    if precision > 0 and recall > 0:
        f1_score = float(2*(precision * recall)/(precision + recall))
    return f1_score

def calculate_recalls(predictions, ground_truths):
    recalls = []
    for i, prediction in enumerate(predictions):
        recall = calculate_recall(prediction, ground_truths[i])
        recalls.append(recall)
    
    return np.mean(recalls)

def calculate_precisions(predictions, ground_truths):
    precisions = []
    for i, prediction in enumerate(predictions):
        precision = calculate_precision(prediction, ground_truths[i])
        precisions.append(precision)
    
    return np.mean(precisions)

def calculate_f1_scores(predictions, ground_truths):
    f1_scores = []
    for i, prediction in enumerate(predictions):
        f1_score = calculate_f1_score(prediction, ground_truths[i])
        f1_scores.append(f1_score)
    
    return np.mean(f1_scores)


predictions = ['putString', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles']
ground_truths = ['putString', 'remove', 'getSoundSource', 'loop', 'pause', 'setLooping', 'add', 'setVolume', 'dispose', 'clear', 'setPitch', 'main']


f1_score = calculate_f1_scores(predictions, ground_truths)
print(f1_score)