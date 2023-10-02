import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Serial

class TransformerRegression:
    def __init__(self, n_layers, n_hidden):
        self.init_fn, self.apply_fn = Serial(*[Dense(n_hidden), Relu] * n_layers, Dense(1))
        self.optimizer = optimizers.adam(1e-3)
        self.state = self.optimizer.init(self.init_fn)[1]

    @jit
    def predict(self, inputs):
        return self.apply_fn(inputs)

    @jit
    def update(self, inputs, targets):
        def loss_fn(params):
            predictions = self.apply_fn(params, inputs)
            return jnp.mean(jnp.square(predictions - targets))
        grad_fn = grad(loss_fn)
        self.state = self.optimizer.update(grad_fn, self.state)
        return self.stateransport



class GNN_link_prediction(predictor_base):
    def __init__(self):
        pass
    @jit
    def predict(self):
        pass



class edge_flow:
    def __init__(self):
        pass

    def fit_edgeFlow(self):
        pass

    def extrapolate_edgeFlow(self):
        pass

    def edge_flowToProb(self):
        pass