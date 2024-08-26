import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, callbacks

from code_loader.experiment_api.experiment import Experiment
from code_loader import init_experiment, Client

#####################################################################
#### This example showcase TL experiment tracking interface use #####
#####################################################################

class LeapTrackCheckpoint(callbacks.Callback):
    """
    A custom Keras callback for tracking and saving model checkpoints during training.

    Attributes:
        train_data (tuple): A tuple containing training data and labels.
        val_data (tuple): A tuple containing validation data and labels.
        leap_tracker (Experiment): An instance of Tensorleap's Experiment class for logging and tracking.
        save_freq (int): Frequency (in epochs) to save the model checkpoint.
        save_path (str): Path template where model checkpoints will be saved.
        save_weights_flag (bool): Whether to save model weights with unique tags per epoch.
    """
    def __init__(self, train_data, val_data, leap_tracker: Experiment,  save_path: str, save_freq: int = 10, save_weights_flag: bool = False):
        """
        Initializes the LeapTrackCheckpoint callback.

        Args:
            train_data (tuple): A tuple containing training data and labels.
            val_data (tuple): A tuple containing validation data and labels.
            leap_tracker (Experiment): An instance of Tensorleap's Experiment class for logging and tracking.
            save_path (str): Path template where model checkpoints will be saved.
            save_freq (int, optional): Frequency (in epochs) to save the model checkpoint. Defaults to 10.
            save_weights_flag (bool, optional): Whether to save model weights with unique tags per epoch. Defaults to False.
        """

        super(LeapTrackCheckpoint, self).__init__()
        self.min_loss = np.inf
        self.train_data = train_data
        self.val_data = val_data
        self.leap_tracker = leap_tracker
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_weights_flag = save_weights_flag

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch during training.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): A dictionary of logs from the training process.
        """
        # Custom behavior: print a message, log something, etc.
        if (epoch + 1) % self.save_freq == 0:
            self.custom_behavior(epoch)

    def on_train_end(self, logs=None):
        """
        Called at the end of the training process.

        Args:
            logs (dict, optional): A dictionary of logs from the training process.
        """
        self.custom_behavior("last", final_epoch=True)

    def custom_behavior(self, epoch, final_epoch=False):
        """
        Custom behavior to save the model and log metrics at the end of an epoch or training.

        Args:
            epoch (int or str): The current epoch index or 'last' for the final epoch.
            final_epoch (bool, optional): Flag indicating if this is the final epoch. Defaults to False.
        """
        if final_epoch:
            file_path = 'model_checkpoint_epoch_last'
            print(f"\nSaving model at the last epoch to {file_path}")
        else:
            print(f"\nEpoch {epoch + 1}: saving model to {self.save_path.format(epoch=epoch + 1)}")
            file_path = self.save_path.format(epoch=epoch + 1)

        # Save the model
        self.model.save(file_path)

        # Eval on train and val sets
        train_loss, train_acc = self.model.evaluate(*self.train_data, verbose=0)
        val_loss, val_acc = self.model.evaluate(*self.val_data, verbose=0)

        tags = ["latest"]
        if val_loss < self.min_loss:
            tags += ["best"]
            self.min_loss = val_loss

        # To be able to save each model weights, we add unique tag name
        if self.save_weights_flag:
            tags += [f"epoch_{epoch}"]

        # Send trial metrics to TL
        print('Sending trial metrics to Tensorleap')
        self.leap_tracker.log_epoch(epoch=epoch, metrics={
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, model_path=file_path, tags=tags)



def main():

    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

    # Downsample for test
    n = 20
    x_train, y_train = x_train[:n], y_train[:n]
    x_test, y_test = x_test[:n], y_test[:n]

    # Add a channel dimension to the data
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    input_shape = (28, 28, 1)

    # Hyper-parameters
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    epochs = 50

    # Client: add auth credentials: default will fetch configured cli auth
    # Working dir: path where leap.yaml is saved
    leap_tracker: Experiment = init_experiment(experiment_name="ExpTrial", project_name="mnist_exp",
                                               code_integration_name="mnist",
                                               description="", client=None)
    # Log experiment metadata properties
    leap_tracker.set_properties({
        "description": "This is an example for TL training tracking",
        "dataset": "mnist",
        "model": "Vanilla CNN",
        "input shape": str(input_shape),
        "subset_sizes": {
            "train": len(x_train),
            "val": len(x_test)},
        "epochs": epochs,
        "optimizer": str(optimizer),
        "loss": str(loss),
        "metrics": metrics,
        })

    # Create a CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Set up the custom checkpoint callback
    checkpoint_callback = LeapTrackCheckpoint(
        train_data=(x_train, y_train),
        val_data=(x_test, y_test),
        leap_tracker=leap_tracker,
        save_freq=10,
        save_path='model_checkpoint_epoch_{epoch:02d}.h5'
    )

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # Train the model
    # Train the model with the custom checkpoint callback
    model.fit(
        x_train, y_train,
        epochs=epochs,  # Train for more epochs to demonstrate checkpointing
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Save the model
    model.save("mnist_cnn_model.h5")



if __name__ == "__main__":
    main()