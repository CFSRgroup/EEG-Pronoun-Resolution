data = load();
time_features = data.time_features;
frequency_features = data.frequency_features;
theta_frequency_features = frequency_features(:,1:14)
alpha_frequency_features = frequency_features(:,15:28)
beta_frequency_features = frequency_features(:,29:42)
gamma_frequency_features = frequency_features(:,43:56)
theta_alpha=[theta_frequency_features alpha_frequency_features]
theta_alpha_beta = [theta_alpha beta_frequency_features]
% time_features = sgolayfilt(time_features, 15, 19);
% frequency_features = sgolayfilt(frequency_features, 15, 19);

labels = readmatrix();
combined_features = [time_features frequency_features];
rng(1);

filtered_features = frequency_features

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

%     % lr
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

[max_accuracy, idx_max] = max(accuracy_results);
fprintf('\nHighest classification accuracy: %.2f%% using %.0f%% of features.\n', max_accuracy * 100, feature_percentages(idx_max)*100);
disp('Confusion Matrix for the highest accuracy:');
disp(confusion_matrices{idx_max});
