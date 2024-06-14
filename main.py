
from data_loader import DataLoader
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator


train_ds, val_ds, test_ds = DataLoader(num_classes=8,
                                       n_frames=4,
                                       data_split={"train": 30, "val": 10, "test": 10},
                                       batch_size=8).load()

model, history = ModelTrainer(train_ds,
                              val_ds,
                              n_frames=4,
                              learning_rate=0.001,
                              epochs=1).train()


ModelEvaluator(model, history, test_ds).evaluate()

