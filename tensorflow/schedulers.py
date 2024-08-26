import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

'''class LinearDecay(LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, decay_steps):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        # return self.initial_learning_rate / (step + 1)
        # Ensure step is within bounds
        step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
        final_lr = tf.cast(self.final_learning_rate, tf.float32)
        
        # Compute the learning rate using linear interpolation
        learning_rate = tf.maximum(
            initial_lr - (initial_lr - final_lr) * (step / decay_steps),
            final_lr
        )
        return learning_rate
    
    def get_config(self):
        # Return the configuration of the learning rate schedule
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'final_learning_rate': self.final_learning_rate,
            'decay_steps': self.decay_steps
        }'''

class LinearDecay(LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, total_steps):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_steps

    def __call__(self, step):
        # return self.initial_learning_rate / (step + 1)
        # Ensure step is within bounds
        step = tf.cast(step, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
        final_lr = tf.cast(self.final_learning_rate, tf.float32)
        
        # Compute the learning rate using linear interpolation
        learning_rate = tf.maximum(
            initial_lr - (initial_lr - final_lr) * (step / total_steps),
            final_lr
        )

        # Print the learning rate for debugging
        # tf.print("Step:", step, "Learning Rate:", learning_rate)

        return learning_rate
    
    def get_config(self):
        # Return the configuration of the learning rate schedule
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'final_learning_rate': self.final_learning_rate,
            'total_steps': self.total_steps
        }