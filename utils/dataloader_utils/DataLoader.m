function [Xtrain, Ytrain, Xtest, Ytest] = DataLoader(filename)

if strcmp(filename, 'mnist')
    filepaths.Xtrain = 'datasets/mnist/train-images-idx3-ubyte.gz';
    filepaths.Ytrain = 'datasets/mnist/train-labels-idx1-ubyte.gz';
    filepaths.Xtest = 'datasets/mnist/t10k-images-idx3-ubyte.gz';
    filepaths.Ytest = 'datasets/mnist/t10k-labels-idx1-ubyte.gz';

    Xtrain = processImagesMNIST(filepaths.Xtrain);
    Ytrain = processLabelsMNIST(filepaths.Ytrain);
    Xtest = processImagesMNIST(filepaths.Xtest);
    Ytest = processLabelsMNIST(filepaths.Ytest);
elseif strcmp(filename, 'cifar10')
    filepath = 'datasets/cifar10';
    [Xtrain, Ytrain, Xtest, Ytest] = loadCIFAR10(filepath);
else
    fprintf('Please enter valid filename.\n');
end


end