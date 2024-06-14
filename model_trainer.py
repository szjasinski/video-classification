import keras
from keras import layers


class ModelTrainer:
    def __init__(self, train_ds, val_ds, n_frames, learning_rate, epochs):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.n_frames = n_frames
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = keras.Sequential([
            layers.Input(shape=(self.n_frames, 224, 224, 3)),

            layers.Conv3D(filters=16, kernel_size=(1, 7, 7), padding="same"),
            layers.Conv3D(filters=16, kernel_size=(3, 1, 1), padding="same"),
            layers.ReLU(),

            layers.MaxPool3D(pool_size=(1, 5, 5)),
            layers.BatchNormalization(),

            layers.Conv3D(filters=32, kernel_size=(1, 3, 3), padding="same"),
            layers.Conv3D(filters=32, kernel_size=(3, 1, 1), padding="same"),
            layers.ReLU(),

            layers.MaxPool3D(pool_size=(1, 5, 5)),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8)
        ])

        self.model.summary()

    def train(self):
        frames, label = next(iter(self.train_ds))
        self.model.build(frames)

        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                           metrics=['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8)

        history = self.model.fit(x=self.train_ds,
                                 epochs=self.epochs,
                                 validation_data=self.val_ds,
                                 callbacks=[early_stopping])

        return self.model, history
