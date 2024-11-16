data = load('D');
time_features = data.time_features;
frequency_features = data.frequency_features;
% time_features = sgolayfilt(time_features, 15, 19);
% frequency_features = sgolayfilt(frequency_features, 15, 19);

labels = readmatrix('D:);

% 合并时域和频域特征以进行特征选择
combined_features = [time_features frequency_features];

K = 50;
[RANKED, WEIGHT] = relieff(combined_features, labels, K);


[~, sorted_indices] = sort(WEIGHT, 'descend');
selected_indices = RANKED(sorted_indices);
rng(1);
feature_percentages = 0.05:0.05:1;
accuracy_results = zeros(length(feature_percentages), 1);
confusion_matrices = cell(length(feature_percentages), 1);

for p = 1:length(feature_percentages)
    num_features_to_select = floor(length(selected_indices) * feature_percentages(p));
    current_selected_indices = selected_indices(1:num_features_to_select);
    filtered_features = combined_features(:, current_selected_indices);
    features_train = [];
    features_test = [];
    labels_train = [];
    labels_test = [];
    for subject = 1:20
        for label = 0:2
            idx = find(labels((subject-1)*200 + 1 : subject*200) == label);
            idx = idx + (  subject-1)*200;
            num_train = floor(length(idx) * 0.7);
            idx = idx(randperm(length(idx)));
            train_idx = idx(1:num_train);
            test_idx = idx(num_train+1:end);
            features_train = [features_train; filtered_features(train_idx, :)];
            labels_train = [labels_train; labels(train_idx)];
            features_test = [features_test; filtered_features(test_idx, :)];
            labels_test = [labels_test; labels(test_idx)];
        end
           %LDA
        model = fitcdiscr(features_train, labels_train);
        predictions = predict(model, features_test);
        accuracy = sum(predictions == labels_test) / numel(labels_test);
        accuracy_results(p) = accuracy;
        fprintf('Using %.0f%% of features, classification accuracy: %.2f%%\n', feature_percentages(p)*100, accuracy * 100);

    end

    %*************************

    % SVM
    %model = fitcecoc(features_train, labels_train, 'Learners', 'svm');

    % KNN
    %K = 50;
    %model = fitcknn(features_train, labels_train, 'NumNeighbors', K);

%     %LDA
%     model = fitcdiscr(features_train, labels_train);

    %model = TreeBagger(100, features_train, labels_train, 'Method', 'classification');

    % 逻辑回归lr
%     template = templateLinear('Learner', 'logistic');
%     model = fitcecoc(features_train, labels_train, 'Learners', template);
    
    %*************************

%     % 使用测试集评估模型
%     predictions = predict(model, features_test);

    % 随机森林  将预测结果从cell转换为适当的格式
    %predictions = str2double(predictions);

%     
%     % 计算准确率
%     accuracy = sum(predictions == labels_test) / numel(labels_test);
%     accuracy_results(p) = accuracy;
%     fprintf('Using %.0f%% of features, classification accuracy: %.2f%%\n', feature_percentages(p)*100, accuracy * 100);


%     % 计算并存储混淆矩阵
%     confMat = confusionmat(labels_test, predictions);
%     confusion_matrices{p} = confMat;
%     fprintf('Confusion Matrix:\n');
%     disp(confMat);
end

[max_accuracy, idx_max] = max(accuracy_results);
fprintf('\nHighest classification accuracy: %.2f%% using %.0f%% of features.\n', max_accuracy * 100, feature_percentages(idx_max)*100);
disp('Confusion Matrix for the highest accuracy:');
disp(confusion_matrices{idx_max});
