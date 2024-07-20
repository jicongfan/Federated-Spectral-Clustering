function [X, Label] = fedsc_dataloader(ds_name)
%
% Args
% ds_name: str, name of dataset
%
% Outputs
% X: m by n, double, m features and n data points
% Labels: n by 1, int, label of data points
%

switch(ds_name)
    case 'CC'
        ds = load('Concentric_Circles.mat');
        X = ds.X;
        Label = ds.Label;
    case 'Iris'
        ds = load('Iris_tsne.mat');
        X = ds.X;
        Label = ds.Label;
    case 'COIL20'
        ds = load('COIL20_tsne.mat');
        X = ds.X;
        Label = ds.Label;
    case 'Bank'
        ds = load('Bank_tsne.mat');
        X = ds.X;
        Label = ds.Label;
    case 'USPS'
        ds = load('USPS_tsne.mat');
        X = ds.X;
        Label = ds.Label;
    case 'ORL'
        ds = load('ORL_tsne.mat');
        X = ds.X;
        Label = ds.Label;
    case 'mnist'
        % class: 10
        % train: 60000
        % test: 10000
        [~, ~, Xtest, Ytest] = DataLoader('mnist');
        mask = [];
        for cname = unique(Ytest)'
            cidx = find(Ytest == cname);
            cmask = randsample(1:length(cidx), 100);
            mask = [mask; cidx(cmask)];
        end
        Xtest = Xtest(:, :, :, mask);
        Ytest = Ytest(mask);
        % X = reshape(cat(4, Xtrain, Xtest), [], 70000);
        % Label = cat(1, double(Ytrain),double(Ytest));
        X = reshape(Xtest, [], 1000);
        Label = double(Ytest);
    case 'cifar10'
        % class: 10
        % train: 50000
        % test: 10000
        [~, ~, Xtest, Ytest] = DataLoader('cifar10');
        Xtest = im2double(Xtest);
        mask = [];
        for cname = unique(Ytest)'
            cidx = find(Ytest == cname);
            cmask = randsample(1:length(cidx), 100);
            mask = [mask; cidx(cmask)];
        end
        Xtest = Xtest(:, :, :, mask);
        Ytest = Ytest(mask);
        % Xtrain = 0.2989 * Xtrain(:,:,1,:) + 0.5870 * Xtrain(:,:,2,:) + 0.1140 * Xtrain(:,:,3,:);
        % Xtest = 0.2989 * Xtest(:,:,1,:) + 0.5870 * Xtest(:,:,2,:) + 0.1140 * Xtest(:,:,3,:);
        % X = reshape(cat(4, Xtrain, Xtest), [], 60000);
        % Label = cat(1, double(Ytrain),double(Ytest));
        X = reshape(Xtest, [], 1000);
        Label = double(Ytest);
    otherwise
        error('Undefined dataset.');
end







end