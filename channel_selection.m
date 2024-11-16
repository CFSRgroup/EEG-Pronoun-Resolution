clc,clear;
data = load();
time_features_initial = data.time_features;
frequency_features_initial = data.frequency_features;

time_features = zeros(4000, 14, 9);
for i = 1:14
    part = time_features_initial(:, (i-1)*9+1:i*9);
    time_features(:, i, :) = part;
end
time_features = permute(time_features, [2,1,3]);

frequency_features = zeros(4000, 14, 4);
for i = 1:4
    part = frequency_features_initial(:, (i-1)*14+1:i*14);
    frequency_features(:, :, i) = part;
end

frequency_features = permute(frequency_features, [2,1,3]);
combined_features_total = cat(3, time_features, frequency_features);

ch_weight_list = [];
ch_acc_list = [];

for sub = 1:14
    combined_features = squeeze(combined_features_total(sub,:,:));
    % time_features = sgolayfilt(time_features, 15, 19);
    % frequency_features = sgolayfilt(frequency_features, 15, 19);

    labels = readmatrix();
%     combined_features = cat(3, time_features, frequency_features);

    K = 50;
    [RANKED, WEIGHT] = relieff(combined_features, labels, K);
    [~, sorted_indices] = sort(WEIGHT, 'descend');
    selected_indices = RANKED(sorted_indices);
    rng(1);
    feature_percentages = 0.10:0.10:1;
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
        %K = 50;
        %model = fitcknn(features_train, labels_train, 'NumNeighbors', K);
    
        %LDA
        model = fitcdiscr(features_train, labels_train);

        %model = TreeBagger(100, features_train, labels_train, 'Method', 'classification');

    %     template = templateLinear('Learner', 'logistic');
    %     model = fitcecoc(features_train, labels_train, 'Learners', template);
    %     
        %*************************
        predictions = predict(model, features_test);
        %predictions = str2double(predictions);
        accuracy = sum(predictions == labels_test) / numel(labels_test);
        accuracy_results(p) = accuracy;
        fprintf('Using %.0f%% of features, classification accuracy: %.2f%%\n', feature_percentages(p)*100, accuracy * 100);

        confMat = confusionmat(labels_test, predictions);
        confusion_matrices{p} = confMat;
        fprintf('Confusion Matrix:\n');
        disp(confMat);

        if p == length(feature_percentages)
            ch_acc_list(end+1) = accuracy*100;
        end
    end
    [max_accuracy, idx_max] = max(accuracy_results);
    fprintf('\nHighest classification accuracy: %.2f%% using %.0f%% of features.\n', max_accuracy * 100, feature_percentages(idx_max)*100);
    disp('Confusion Matrix for the highest accuracy:');
    disp(confusion_matrices{idx_max});

    ch_weight_total = sum(WEIGHT(:));
    ch_weight_list(end+1) = ch_weight_total;

end
disp('权重列表的所有内容:');
disp(ch_weight_list);

[maxValue, maxIndex] = max(ch_weight_list);

fprintf('列表中的最大值为: %d\n', maxValue);
fprintf('最大值的索引号为: %d\n', maxIndex);

disp('精度列表的所有内容:');
disp(ch_acc_list);
[maxValue, maxIndex] = max(ch_acc_list);

fprintf('列表中的最大值为: %d\n', maxValue);
fprintf('最大值的索引号为: %d\n', maxIndex);

A_weight = ch_weight_list;
[~, rank] = sort(A_weight, 'descend');

combined_features_total_copy = zeros(14, 4000, 13);

for i = 1:length(rank)
    indice = rank(i);
    combined_features_total_copy(i,:,:)= combined_features_total(indice,:,:);
end

scale = [3, 6, 9, 12, 14];
acc_list_scale = [];


for i = scale
    feature_new = combined_features_total_copy(1:i,:,:);
    feature_new = permute(feature_new,[2,1,3]);
    feature_new = reshape(feature_new, [size(feature_new, 1), numel(feature_new) / size(feature_new, 1)]);
    labels = readmatrix();
    acc_list_scale_sub = [];
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
            features_train = [features_train; feature_new(train_idx, :)];
            labels_train = [labels_train; labels(train_idx)];
            features_test = [features_test; feature_new(test_idx, :)];
            labels_test = [labels_test; labels(test_idx)];
        end
    end
    %LDA
    model = fitcdiscr(features_train, labels_train);
    predictions = predict(model, features_test);
    accuracy = sum(predictions == labels_test) / numel(labels_test);
    acc_list_scale(end+1)= accuracy;
end
disp('scale精度列表的所有内容:');
disp(acc_list_scale);