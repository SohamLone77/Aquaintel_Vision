"""
Simple loss functions for image enhancement
"""
import tensorflow as tf

class SimpleLosses:
    """Collection of basic loss functions"""
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error Loss"""
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    @staticmethod
    def mae_loss(y_true, y_pred):
        """Mean Absolute Error Loss"""
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    @staticmethod
    def ssim_loss(y_true, y_pred, max_val=1.0):
        """Structural Similarity Loss"""
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    
    @staticmethod
    def combined_loss(y_true, y_pred, alpha=0.5):
        """Combine MSE and SSIM losses"""
        mse = SimpleLosses.mse_loss(y_true, y_pred)
        ssim = SimpleLosses.ssim_loss(y_true, y_pred)
        return mse + alpha * ssim
