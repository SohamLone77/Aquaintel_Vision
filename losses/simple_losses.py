"""Simple loss functions for image enhancement with edge preservation."""

import tensorflow as tf


class SimpleLosses:
    """Collection of loss functions for sharp image enhancement."""

    # Backward-compatible tuning knob used by the existing training pipeline.
    combined_alpha = 0.5

    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error Loss - pixel level accuracy."""
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def mae_loss(y_true, y_pred):
        """Mean Absolute Error Loss - more robust to outliers."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    @staticmethod
    def ssim_loss(y_true, y_pred, max_val=1.0):
        """Structural Similarity Loss - preserves structure."""
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))

    @staticmethod
    def edge_loss(y_true, y_pred):
        """Edge preservation loss using Sobel gradients on grayscale images."""
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)

        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

        y_true_gray = tf.image.rgb_to_grayscale(y_true)
        y_pred_gray = tf.image.rgb_to_grayscale(y_pred)

        edges_true_x = tf.nn.conv2d(y_true_gray, sobel_x, strides=[1, 1, 1, 1], padding="SAME")
        edges_true_y = tf.nn.conv2d(y_true_gray, sobel_y, strides=[1, 1, 1, 1], padding="SAME")
        edges_true = tf.sqrt(edges_true_x ** 2 + edges_true_y ** 2 + 1e-6)

        edges_pred_x = tf.nn.conv2d(y_pred_gray, sobel_x, strides=[1, 1, 1, 1], padding="SAME")
        edges_pred_y = tf.nn.conv2d(y_pred_gray, sobel_y, strides=[1, 1, 1, 1], padding="SAME")
        edges_pred = tf.sqrt(edges_pred_x ** 2 + edges_pred_y ** 2 + 1e-6)

        return tf.reduce_mean(tf.abs(edges_true - edges_pred))

    @staticmethod
    def gradient_loss(y_true, y_pred):
        """Gradient difference loss for sharper local detail."""
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]

        loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
        loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
        return loss_x + loss_y

    @staticmethod
    def combined_loss(y_true, y_pred, alpha=None, beta=None, gamma=None):
        """
        Combined loss with backward compatibility.

        Legacy mode (existing training code): if beta/gamma are None,
        this returns `mse + alpha * ssim` where alpha defaults to
        `SimpleLosses.combined_alpha`.

        Extended mode: if alpha, beta, gamma are all provided, this returns
        `alpha * mse + beta * ssim + gamma * edge`.
        """
        mse = SimpleLosses.mse_loss(y_true, y_pred)
        ssim = SimpleLosses.ssim_loss(y_true, y_pred)

        if beta is None or gamma is None:
            legacy_alpha = SimpleLosses.combined_alpha if alpha is None else alpha
            return mse + legacy_alpha * ssim

        edge = SimpleLosses.edge_loss(y_true, y_pred)
        effective_alpha = 0.6 if alpha is None else alpha
        return effective_alpha * mse + beta * ssim + gamma * edge

    @staticmethod
    def sharp_loss(y_true, y_pred):
        """Specialized sharpness-focused loss for underwater enhancement."""
        mse = SimpleLosses.mse_loss(y_true, y_pred)
        edge = SimpleLosses.edge_loss(y_true, y_pred)
        grad = SimpleLosses.gradient_loss(y_true, y_pred)
        return 0.4 * mse + 0.4 * edge + 0.2 * grad
