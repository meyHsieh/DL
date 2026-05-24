from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(loc=0.0, scale=scale, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        """
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        """
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = X @ self.params['W'] + self.params['b'] 
        return output

    def backward(self, grad : np.ndarray):
        self.grads['W'] = self.input.T @ grad  
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k = kernel_size
        self.stride = stride
        self.padding = padding

        n_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.normal(loc=0.0, scale=scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))
        
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self, X):
        self.input = X
        batch, C, H, W = X.shape
        k = self.kernel_size
        out_c = self.out_channels
        
        out_h = (H - k) // self.stride + 1
        out_w = (W - k) // self.stride + 1
        
        i0 = np.repeat(np.arange(k), k)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = np.tile(np.arange(k), k * C).reshape(-1, 1) + j1.reshape(1, -1)
        k_idx = np.repeat(np.arange(C), k * k).reshape(-1, 1)
        
        self.X_col = X[:, k_idx, i, j].transpose(1, 2, 0).reshape(C * k * k, -1)
        
        W_col = self.params['W'].reshape(out_c, -1)
        out = W_col @ self.X_col + self.params['b'] 

        out = out.reshape(out_c, out_h, out_w, batch).transpose(3, 0, 1, 2)
        return out

    def backward(self, grad):
        batch, out_c, out_h, out_w = grad.shape
        C, k = self.in_channels, self.kernel_size

        grad_col = grad.transpose(1, 2, 3, 0).reshape(out_c, -1)

        self.grads['W'] = (grad_col @ self.X_col.T).reshape(out_c, C, k, k)
        self.grads['b'] = np.sum(grad_col, axis=1, keepdims=True)

        W_col = self.params['W'].reshape(out_c, -1)
        dX_col = W_col.T @ grad_col 

        dX = np.zeros_like(self.input)
 
        i0 = np.repeat(np.arange(k), k)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = np.tile(np.arange(k), k * C).reshape(-1, 1) + j1.reshape(1, -1)
        k_idx = np.repeat(np.arange(C), k * k).reshape(-1, 1)

        k_idx_long = np.broadcast_to(k_idx, i.shape).ravel()
        i_long = i.ravel()
        j_long = j.ravel()

        dX_reshaped = dX_col.reshape(C*k*k, out_h*out_w, batch).transpose(2, 0, 1)
        
        for b in range(batch):
            np.add.at(dX[b], (k_idx_long, i_long, j_long), dX_reshaped[b].ravel())
            
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}
        

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
    

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model            
        self.max_classes = max_classes
        self.has_softmax = True       
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.probs = softmax(predicts)

        correct_probs = self.probs[np.arange(len(labels)), labels]
        loss = -np.mean(np.log(correct_probs + 1e-10))
        return loss
    
    def backward(self):
        batch_size = len(self.labels)
        num_classes = self.probs.shape[1]

        self.grads = self.probs.copy() 
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads = self.grads / batch_size
        
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition