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

bestAccuracy = 0;
bestNSel = 0;

for ratio = 0.05:0.05:1
    nSel = floor(182 * ratio);
    [Y, L, V, Label] = DGUFS(X, nClass, S, alpha, beta, nSel);
    selected_features = Y';
    filtered_features = selected_features;
    rng(1);
    features_train = [];
    features_test = [];
    labels_train = [];
    labels_test = [];

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

    %***********************

    % SVM
    %model = fitcecoc(features_train, labels_train, 'Learners', 'svm');

    % KNN
    K = 50;
    model = fitcknn(features_train, labels_train, 'NumNeighbors', K);

    % LDA
    %model = fitcdiscr(features_train, labels_train);
    model = fitcdiscr(features_train, labels_train, 'DiscrimType', 'pseudoLinear');

    %model = TreeBagger(100, features_train, labels_train, 'Method', 'classification');

    % lr
    %template = templateLinear('Learner', 'logistic');
    %model = fitcecoc(features_train, labels_train, 'Learners', template);

    %***********************
    predictions = predict(model, features_test);
    %predictions = str2double(predictions);
    accuracy = sum(predictions == labels_test) / numel(labels_test);

    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestNSel = nSel;
    end
end

% 打印最高的准确率及其对应的nSel
fprintf('Best Classification Accuracy: %.2f%% with nSel: %.0f\n', bestAccuracy * 100, bestNSel);
