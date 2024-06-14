from data_loader import DataLoader
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

if __name__ == "__main__":
    train_ds, val_ds, test_ds = DataLoader(num_classes=8,
                                           n_frames=4,
                                           data_split={"train": 30, "val": 10, "test": 10},
                                           batch_size=8).load()

    model, history = ModelTrainer(train_ds=train_ds,
                                  val_ds=val_ds,
                                  n_frames=4,
                                  learning_rate=0.001,
                                  epochs=50).train()

    ModelEvaluator(model=model,
                   history=history,
                   test_ds=test_ds).evaluate()
