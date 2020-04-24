clc ; close all ; clear all;
%% train
trainfolder= fullfile('C:\Users\BAYSAL\Desktop\tasarým2\fer_orj\fer_orj\train');
categories ={'0', '1', '2',...
    '3','4','5','6',};
TrainingSet= imageDatastore(fullfile(trainfolder,categories),...
    'IncludeSubfolders',true,'Labelsource','foldernames');
%% testing
testfolder= fullfile('C:\Users\BAYSAL\Desktop\tasarým2\fer_orj\fer_orj\test');
categories ={'0', '1', '2',...
    '3','4','5','6',};
testing= imageDatastore(fullfile(testfolder,categories),...
    'IncludeSubfolders',true,'Labelsource','foldernames');
t1=countEachLabel(TrainingSet);
t2=countEachLabel(testing) ;
% [TrainingSet,testing]= splitEachLabel(imds,0.7,'randomize');
imagesize=[48 48 1];
imageAugmenter =imageDataAugmenter('RandXReflection',true);
augimdsTrain = augmentedImageDatastore(imagesize(1:2),TrainingSet,...
    'DataAugmentation',imageAugmenter);
%% layer
layers = [
    imageInputLayer(imagesize)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    maxPooling2dLayer(1,'Stride',1)
    
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    maxPooling2dLayer(1,'Stride',1)
    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

miniBatchSize = 128;
valFrequency = floor(numel(TrainingSet.Files)/miniBatchSize);
%% options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',35, ...
    'MiniBatchSize',128, ...
    'L2Regularization',1e-6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testing, ...
    'ValidationFrequency',valFrequency, ...
    'ValidationPatience', Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,layers,options);
yred = classify(net,testing);
Yvalidation = testing.Labels;
Acc = sum(yred == Yvalidation)/numel(Yvalidation);
save net
plotconfusion(Yvalidation,yred);
