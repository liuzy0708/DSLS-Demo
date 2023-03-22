# Imports
import warnings
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from Define_DSA_AI import DSA_AI_strategy

warnings.filterwarnings("ignore")

def memory_func(n_class, X_memory, y_memory):
    X_memory_collection = [[]] * n_class
    y_memory_collection = [[]] * n_class

    for cls in range(n_class):
        for sample in range(y_memory.shape[0]):
            if y_memory[sample] == cls:
                y_memory_collection[cls] = y_memory_collection[cls] + [cls]
                X_memory_collection[cls] = X_memory_collection[cls] + [X_memory[sample, :]]
    return X_memory_collection, y_memory_collection

n_class = 2
n_round = 1
n_inital = 200

# stream = WaveformGenerator(random_state=1) #3 class
stream = SEAGenerator(random_state=1) #2 class

X_memory, y_memory = stream.next_sample(n_inital)
X_memory_collection, y_memory_collection = memory_func(n_class, X_memory, y_memory)

DSLS_DSA_AI_str = DSA_AI_strategy(memory_collection=y_memory_collection, X_memory_collection=X_memory_collection, d=X_memory.shape[1], n_class=n_class)

max_samples = 1000000  # The range of tested stream

for main_loop in range(n_round):

    # Setup Hyper-parameters
    correct_cnt = 0
    result = []
    y_pred = 0
    count = 0
    n_correct = 0
    result_pred = []
    isLabel_collection = []

    # Setup Classifier
    clf = AdaptiveRandomForestClassifier()

    clf.fit(X_memory, y_memory)

    # Train the classifier with the samples provided by the data stream
    while count < max_samples and stream.has_more_samples():

        X, y = stream.next_sample()

        y_pred = clf.predict(X)
        if y_pred[0] == y:
            n_correct += 1
        result = result + [y[0]]
        result_pred = result_pred + [y_pred[0]]

        clf, isLabel = DSLS_DSA_AI_str.DSA_AI_evaluation(X, y, clf)
        isLabel_collection = isLabel_collection + [isLabel]

        count += 1

        if count % (max_samples * 0.10) == 0:
            print('\nHave processed {:.0f}%'.format(count / max_samples * 100), 'samples')


#Display Results
print("\nAnnotation Rate", sum(isLabel_collection) / max_samples)
print("Overall_Accuracy", n_correct / max_samples)

