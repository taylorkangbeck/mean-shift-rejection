import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='ResNets with Mean Shift Rejection (mxnet)')
    return parser.parse_args()


def main():
    args = parse_args()
    ...


def train_one_epoch():
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # TODO APPLY GRADIENT MODIFICATIONS
        apply_gradient_mods()

        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in valid_data:
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data),
        valid_acc / len(valid_data), time.time() - tic))


def train():
    for epoch in range(10):
        train_one_epoch()


def apply_gradient_mods():
    if ZMG > 0.:
        zero_mean_norm_grads()


def zero_mean_norm_grads(params, per_channel=False, mean_mult=10.0):
        ### DO NO PORT
        if per_channel:  # PerChannel should always be 2
            MeanDims = (2, 3)  # Keep eye out for mxnet dim order
        else:
            MeanDims = (1, 2, 3)
        ###

        # only gradient normalise conv weight params if they have (h,w) spatial dimensions
        for p in params:
            param = p[1]
            param_mean_grad = torch.mean(param.grad, dim=MeanDims, keepdim=True)  # IMPORTANT
            param.grad = param.grad + ( param.grad - param_mean_grad ) * mean_mult


if __name__ == '__main__':
    main()