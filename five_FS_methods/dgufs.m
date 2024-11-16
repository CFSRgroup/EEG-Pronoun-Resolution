data = load('C:\Users\15234\Desktop\FS_test\feature.mat');
time_features = data.time_features;
frequency_features = data.frequency_features;
% sg滤波
% time_features = sgolayfilt(time_features, 15, 19);
% frequency_features = sgolayfilt(frequency_features, 15, 19);

labels = readmatrix('C:\Users\15234\Desktop\FS_test\label.csv', 'OutputType', 'double');

combined_features = [time_features frequency_features];

X = combined_features';
S = X'*X; 


nClass = 3;
alpha = 0.1;
beta = 0.1;
nSel = 100;


[Y, L, V, Label] = DGUFS(X, nClass, S, alpha, beta, nSel);
selected_features = Y';
filtered_features = selected_features;

rng(1);

features_train = [];
features_test = [];
labels_train = [];
labels_test = [];

rng(1);

for subject = 1:20
    for label = 0:2
        idx = find(labels((subject-1)*200 + 1 : subject*200) == label);
        idx = idx + (subject-1)*200;
        num_train = floor(length(idx) * 0.7);
        idx = idx(randperm(length(idx)));
        train_idx = idx(1:num_train);
        test_idx = idx(num_train+1:end);
        features_train = [features_train; filtered_features(train_idx, :)];
        labels_train = [labels_train; labels(train_idx)];
        features_test = [features_test; filtered_features(test_idx, :)];
        labels_test = [labels_test; labels(test_idx)];
    end
end

%*************************

    % SVM
    %model = fitcecoc(features_train, labels_train, 'Learners', 'svm');

    % KNN
    %K = 100;
    %model = fitcknn(features_train, labels_train, 'NumNeighbors', K);

    % LDA
    %model = fitcdiscr(features_train, labels_train);
    model = fitcdiscr(features_train, labels_train, 'DiscrimType', 'pseudoLinear');


    % 随机森林
    %model = TreeBagger(100, features_train, labels_train, 'Method', 'classification');

    % 使用逻辑回归进行三分类
    %template = templateLinear('Learner', 'logistic');
    %model = fitcecoc(features_train, labels_train, 'Learners', template);
    
    %*************************

    % 使用测试集评估模型
    predictions = predict(model, features_test);

    %predictions = str2double(predictions);

    accuracy = sum(predictions == labels_test) / numel(labels_test);
    fprintf('Classification accuracy: %.2f%%\n', accuracy * 100);

    confMat = confusionmat(labels_test, predictions);
    confusion_matrices{p} = confMat;
    fprintf('Confusion Matrix:\n');
    disp(confMat);


