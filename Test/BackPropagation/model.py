import numpy as np


class FullyConnectionModel:
    def __init__(self, model_cfg, flatten):
        self.model_cfg = model_cfg
        self.layers = list()
        in_channels = flatten
        for deep in model_cfg:
            layer = FcLayer(in_channels=in_channels, out_channels=deep)
            self.layers.append(layer)
            in_channels = deep

    def __call__(self, image):
        self.batch_size = image.shape[0]
        # without batch
        # output = image.reshape(-1)[None, :]

        # with batch
        output = image.reshape(self.batch_size, -1)[:, None, :]
        for layer in self.layers:
            output = layer(output)
        return output

    def back_propagate(self, results, targets, lr):
        sigma = list()
        # without batch
        # for result, target in zip(results, targets):
        #     current = (target - result) * result * (1 - result)
        #     sigma.append(current)
        # sigma = np.array(sigma)

        # with batch
        results = results.transpose(1, 0)
        targets = targets.transpose(1, 0)
        for result, target in zip(results, targets):
            current = (target - result) * result * (1 - result)
            sigma.append(current)
        sigma = np.array(sigma).transpose(1, 0)

        for layer in self.layers[::-1]:
            sigma = layer.back_propagate(sigma, lr)


class FcLayer:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.normal(size=(self.in_channels, self.out_channels))
        # self.weights = np.random.random(size=(self.in_channels, self.out_channels))
        self.input = None
        self.output = None

    def __call__(self, x):
        self.batch_size = x.shape[0]
        self.input = x
        # with batch
        extend_weights = np.expand_dims(self.weights, axis=0).repeat(self.batch_size, axis=0)
        output = np.einsum('ijk,ikn->ijn', x, extend_weights)

        # without batch
        # output = np.dot(x, self.weights)
        output = 1 / (1 + np.exp(-output))
        self.output = output
        return output

    def back_propagate(self, sigma, lr):
        # without batch
        # sigma_extend = sigma[None, :]
        # input_extend = self.input.squeeze(0)[:, None]
        # delta_weights = np.matmul(input_extend, sigma_extend) * lr
        # new_weights = self.weights + delta_weights
        # transpose_weights = self.weights.T
        # inputs = self.input.squeeze(0)
        # pos_inputs = 1 - inputs
        # avg = inputs * pos_inputs
        # new_sigma = np.matmul(sigma_extend, transpose_weights)
        # new_sigma = new_sigma * avg
        # self.weights = new_weights
        # return new_sigma.squeeze(0)

        # with batch
        sigma_extend = sigma[:, None, :]
        input_extend = self.input.squeeze(axis=1)[..., None]
        delta_weights = np.einsum('ijk,ikn->ijn', input_extend, sigma_extend) * lr
        delta_weights_mean = delta_weights.mean(axis=0)
        # delta_weights = np.matmul(input_extend, sigma_extend) * lr
        new_weights = self.weights + delta_weights_mean
        transpose_weights = np.expand_dims(self.weights.T, axis=0).repeat(self.batch_size, axis=0)
        inputs = self.input.squeeze(axis=1)
        pos_inputs = 1 - inputs
        avg = inputs * pos_inputs
        avg = avg[:, None, :]
        new_sigma = np.einsum('ijk,ikn->ijn', sigma_extend, transpose_weights)
        new_sigma = new_sigma * avg
        self.weights = new_weights
        return new_sigma.squeeze(axis=1)
