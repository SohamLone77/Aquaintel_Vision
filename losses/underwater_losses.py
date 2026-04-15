


"""
Specialized loss functions for underwater image enhancement
Fully fixed version with correct data types
"""

import tensorflow as tf
import numpy as np

class UnderwaterLosses:
    """
    Collection of loss functions optimized for underwater image enhancement
    """
    
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
    def psnr_loss(y_true, y_pred, max_val=1.0):
        """PSNR-based loss (negative PSNR)"""
        psnr = tf.image.psnr(y_true, y_pred, max_val=max_val)
        return -tf.reduce_mean(psnr)
    
    @staticmethod
    def perceptual_loss(y_true, y_pred):
        """
        Perceptual loss using pre-trained VGG16 features
        Helps preserve high-level features
        """
        # Load VGG16 without top layers
        vgg = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(None, None, 3)
        )
        
        # Use intermediate layers for feature extraction
        layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']
        feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=[vgg.get_layer(layer).output for layer in layers]
        )
        
        # Freeze VGG weights
        feature_extractor.trainable = False
        
        # Get features
        features_true = feature_extractor(y_true * 255)  # VGG expects [0,255]
        features_pred = feature_extractor(y_pred * 255)
        
        # Compute L2 loss between features
        perceptual_loss = 0.0
        for f_true, f_pred in zip(features_true, features_pred):
            perceptual_loss += tf.reduce_mean(tf.square(f_true - f_pred))
        
        return perceptual_loss / float(len(layers))
    
    @staticmethod
    def color_loss(y_true, y_pred):
        """
        Color consistency loss for underwater images
        Penalizes color casts and preserves color distribution
        """
        # Convert to YUV color space
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)
        
        # Split channels
        true_y, true_u, true_v = tf.split(y_true_yuv, 3, axis=-1)
        pred_y, pred_u, pred_v = tf.split(y_pred_yuv, 3, axis=-1)
        
        # Color loss - focus on U and V channels (color information)
        color_diff = tf.reduce_mean(tf.square(true_u - pred_u)) + \
                     tf.reduce_mean(tf.square(true_v - pred_v))
        
        # Add histogram matching loss for better color distribution
        hist_loss = UnderwaterLosses.histogram_loss(y_true, y_pred)
        
        return color_diff + 0.1 * hist_loss
    
    @staticmethod
    def histogram_loss(y_true, y_pred, bins=64):
        """
        Histogram matching loss to preserve overall intensity distribution
        Returns float32 tensor
        """
        # Ensure bins is int32
        bins = tf.cast(bins, tf.int32)
        
        # Flatten the images
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute histograms - nbins must be int32
        hist_true = tf.histogram_fixed_width(
            y_true_flat, 
            [0.0, 1.0], 
            nbins=bins,
            dtype=tf.int32
        )
        hist_pred = tf.histogram_fixed_width(
            y_pred_flat, 
            [0.0, 1.0], 
            nbins=bins,
            dtype=tf.int32
        )
        
        # Convert to float32 for calculations
        hist_true = tf.cast(hist_true, tf.float32)
        hist_pred = tf.cast(hist_pred, tf.float32)
        
        # Normalize histograms
        hist_true = hist_true / (tf.reduce_sum(hist_true) + 1e-7)
        hist_pred = hist_pred / (tf.reduce_sum(hist_pred) + 1e-7)
        
        # Earth Mover's Distance approximation
        cum_true = tf.cumsum(hist_true)
        cum_pred = tf.cumsum(hist_pred)
        
        return tf.reduce_mean(tf.abs(cum_true - cum_pred))
    
    @staticmethod
    def edge_loss(y_true, y_pred):
        """
        Edge preservation loss using Sobel filters
        Important for underwater images where edges can be blurred
        """
        # Sobel filters for edge detection
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Convert to grayscale
        y_true_gray = tf.image.rgb_to_grayscale(y_true)
        y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
        
        # Apply Sobel filters
        edges_true_x = tf.nn.conv2d(y_true_gray, sobel_x, strides=[1,1,1,1], padding='SAME')
        edges_true_y = tf.nn.conv2d(y_true_gray, sobel_y, strides=[1,1,1,1], padding='SAME')
        edges_true = tf.sqrt(edges_true_x**2 + edges_true_y**2 + 1e-6)
        
        edges_pred_x = tf.nn.conv2d(y_pred_gray, sobel_x, strides=[1,1,1,1], padding='SAME')
        edges_pred_y = tf.nn.conv2d(y_pred_gray, sobel_y, strides=[1,1,1,1], padding='SAME')
        edges_pred = tf.sqrt(edges_pred_x**2 + edges_pred_y**2 + 1e-6)
        
        # Edge loss
        return tf.reduce_mean(tf.abs(edges_true - edges_pred))
    
    @staticmethod
    def gradient_loss(y_true, y_pred):
        """
        Gradient difference loss for sharper results
        """
        # Simple gradient difference
        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        
        loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
        loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
        
        return loss_x + loss_y
    
    @staticmethod
    def tv_loss(y_pred):
        """
        Total Variation loss for smoothness
        Useful as a regularization term
        """
        x_diff = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        y_diff = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        return tf.reduce_mean(tf.abs(x_diff)) + tf.reduce_mean(tf.abs(y_diff))

class CombinedUnderwaterLoss:
    """
    Combined loss function with configurable weights for underwater enhancement
    """
    
    def __init__(self, weights=None):
        """
        Initialize with custom weights for each loss component
        
        Args:
            weights: Dictionary with loss weights
                   e.g., {'mse': 1.0, 'ssim': 0.5, 'color': 0.3, 'edge': 0.2}
        """
        if weights is None:
            # Default weights optimized for underwater images
            self.weights = {
                'mse': 1.0,
                'ssim': 0.5,
                'color': 0.3,
                'edge': 0.2,
                'perceptual': 0.1,
                'gradient': 0.1,
                'tv': 0.01
            }
        else:
            self.weights = weights
    
    def __call__(self, y_true, y_pred):
        """Compute combined loss"""
        total_loss = 0.0
        
        # MSE loss
        if 'mse' in self.weights and self.weights['mse'] > 0:
            mse = UnderwaterLosses.mse_loss(y_true, y_pred)
            total_loss += self.weights['mse'] * mse
        
        # MAE loss
        if 'mae' in self.weights and self.weights['mae'] > 0:
            mae = UnderwaterLosses.mae_loss(y_true, y_pred)
            total_loss += self.weights['mae'] * mae
        
        # SSIM loss
        if 'ssim' in self.weights and self.weights['ssim'] > 0:
            ssim = UnderwaterLosses.ssim_loss(y_true, y_pred)
            total_loss += self.weights['ssim'] * ssim
        
        # Color loss
        if 'color' in self.weights and self.weights['color'] > 0:
            color = UnderwaterLosses.color_loss(y_true, y_pred)
            total_loss += self.weights['color'] * color
        
        # Edge loss
        if 'edge' in self.weights and self.weights['edge'] > 0:
            edge = UnderwaterLosses.edge_loss(y_true, y_pred)
            total_loss += self.weights['edge'] * edge
        
        # Perceptual loss
        if 'perceptual' in self.weights and self.weights['perceptual'] > 0:
            perceptual = UnderwaterLosses.perceptual_loss(y_true, y_pred)
            total_loss += self.weights['perceptual'] * perceptual
        
        # Gradient loss
        if 'gradient' in self.weights and self.weights['gradient'] > 0:
            gradient = UnderwaterLosses.gradient_loss(y_true, y_pred)
            total_loss += self.weights['gradient'] * gradient
        
        # TV loss (only on prediction for smoothness)
        if 'tv' in self.weights and self.weights['tv'] > 0:
            tv = UnderwaterLosses.tv_loss(y_pred)
            total_loss += self.weights['tv'] * tv
        
        return total_loss

def create_loss_function(loss_type='combined'):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'mse', 'mae', 'ssim', 'combined', 'perceptual', 
                  'underwater', 'underwater_full', or 'custom'
    
    Returns:
        Loss function
    """
    if loss_type == 'mse':
        return UnderwaterLosses.mse_loss
    elif loss_type == 'mae':
        return UnderwaterLosses.mae_loss
    elif loss_type == 'ssim':
        return UnderwaterLosses.ssim_loss
    elif loss_type == 'combined':
        return lambda y_true, y_pred: (
            UnderwaterLosses.mse_loss(y_true, y_pred) + 
            0.5 * UnderwaterLosses.ssim_loss(y_true, y_pred)
        )
    elif loss_type == 'perceptual':
        return lambda y_true, y_pred: (
            UnderwaterLosses.mse_loss(y_true, y_pred) + 
            0.1 * UnderwaterLosses.perceptual_loss(y_true, y_pred)
        )
    elif loss_type == 'underwater':
        # Basic underwater loss
        weights = {
            'mse': 1.0,
            'ssim': 0.5,
            'color': 0.3,
            'edge': 0.2
        }
        return CombinedUnderwaterLoss(weights)
    elif loss_type == 'underwater_full':
        # Full underwater loss with all components
        weights = {
            'mse': 1.0,
            'ssim': 0.5,
            'color': 0.3,
            'edge': 0.2,
            'perceptual': 0.1,
            'gradient': 0.1,
            'tv': 0.01
        }
        return CombinedUnderwaterLoss(weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Test the losses
if __name__ == "__main__":
    print("="*60)
    print("TESTING UNDERWATER LOSS FUNCTIONS")
    print("="*60)
    
    # Create dummy data
    batch_size = 2
    img_size = 128
    y_true = tf.random.uniform([batch_size, img_size, img_size, 3], dtype=tf.float32)
    y_pred = tf.random.uniform([batch_size, img_size, img_size, 3], dtype=tf.float32)
    
    # Test basic losses
    print("\n📊 Basic Losses:")
    basic_losses = ['mse', 'mae', 'ssim', 'psnr']
    for loss_name in basic_losses:
        loss_func = getattr(UnderwaterLosses, f"{loss_name}_loss")
        loss_value = loss_func(y_true, y_pred)
        print(f"   {loss_name.upper():10}: {loss_value:.4f}")
    
    # Test specialized losses
    print("\n🌊 Specialized Underwater Losses:")
    specialized = ['color', 'edge', 'gradient', 'tv']
    for loss_name in specialized:
        if loss_name == 'tv':
            loss_value = UnderwaterLosses.tv_loss(y_pred)
        else:
            loss_func = getattr(UnderwaterLosses, f"{loss_name}_loss")
            loss_value = loss_func(y_true, y_pred)
        print(f"   {loss_name.capitalize():10}: {loss_value:.4f}")
    
    # Test combined losses
    print("\n🔗 Combined Losses:")
    
    # Simple combined
    simple_combined = lambda y_true, y_pred: (
        UnderwaterLosses.mse_loss(y_true, y_pred) + 
        0.5 * UnderwaterLosses.ssim_loss(y_true, y_pred)
    )
    loss_value = simple_combined(y_true, y_pred)
    print(f"   Simple Combined: {loss_value:.4f}")
    
    # Underwater combined
    underwater_loss = CombinedUnderwaterLoss()
    loss_value = underwater_loss(y_true, y_pred)
    print(f"   Underwater Combined: {loss_value:.4f}")
    
    # Test loss factory
    print("\n🏭 Loss Factory:")
    for loss_type in ['mse', 'ssim', 'combined', 'underwater']:
        loss_fn = create_loss_function(loss_type)
        if callable(loss_fn):
            loss_value = loss_fn(y_true, y_pred)
            print(f"   {loss_type.capitalize():12}: {loss_value:.4f}")
    
    print("\n✅ All loss functions tested successfully!")