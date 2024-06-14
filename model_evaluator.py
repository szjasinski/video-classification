from utilities.evaluator_utilities import plot_history
from utilities.evaluator_utilities import get_actual_predicted_labels
from utilities.evaluator_utilities import plot_confusion_matrix
from utilities.evaluator_utilities import calculate_classification_metrics


class ModelEvaluator:
    def __init__(self, model, history, test_ds):
        self.labels = ["ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
                       "BalanceBeam", "BandMarching", "BaseballPitch", "BasketballDunk"]

        self.model = model
        self.history = history
        self.test_ds = test_ds

    def evaluate(self):
        # accuracy and loss on test set
        evaluation = self.model.evaluate(self.test_ds, return_dict=True)
        print(f"Accuracy on test set: {round(evaluation['accuracy'], 4)}")
        print(f"Loss on test set: {round(evaluation['loss'], 4)}")
        print("")

        # plot accuracy and loss learning curves
        plot_history(self.history)

        # get actual labels from test set and labels predicted by model
        actual, predicted = get_actual_predicted_labels(self.model, self.test_ds)
        plot_confusion_matrix(actual, predicted, self.labels)

        precision, recall = calculate_classification_metrics(actual, predicted, self.labels)

        # display precision and recall for particular classes
        print("Precision:")
        for action, value in precision.items():
            print(action, round(value, 4))

        print("\nRecall:")
        for action, value in recall.items():
            print(action, round(value, 4))
