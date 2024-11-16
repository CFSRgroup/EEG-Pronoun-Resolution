clc;
clear all;
data = load();
time_features = data.time_features;
frequency_features = data.frequency_features;

labels = readmatrix();

combined_features = [time_features frequency_features];

max_accuracy_per_sub = zeros(20, 1);
best_feature_percentage_per_sub = zeros(20, 1);

for sub = 1:20
    new_features = combined_features((sub-1)*200+1:200*sub,:);
    new_labels = labels((sub-1)*200+1:200*sub,:);
    X = new_features';
    S = X'*X;
    sub_accuracy_results = zeros(20, 1);
    index = 1;

    for ratio = 0.05:0.05:1
        nSel = floor(182 * ratio);
        [Y, L, V, Label] = DGUFS(X, 3, S, 0.1, 0.1, nSel);
        selected_features = Y';
        filtered_features = selected_features;
        train_size = floor(0.7 * size(filtered_features, 1));
        test_size = floor(0.3 * size(filtered_features, 1));
        X_train = filtered_features(1:train_size,:);
        X_test = filtered_features(end - test_size + 1:end,:);
        y_train = new_labels(1:train_size,:);
        y_test = new_labels(end - test_size + 1:end,:);
        
        % LDA
        model = fitcdiscr(X_train, y_train,'DiscrimType', 'pseudoLinear');
        predictions = predict(model, X_test);
        accuracy = sum(predictions == y_test) / numel(y_test);
        sub_accuracy_results(index) = accuracy;
        index = index + 1;
    end
    [max_accuracy, best_index] = max(sub_accuracy_results);
    best_feature_percentage = 0.05 * (best_index );
    max_accuracy_per_sub(sub) = max_accuracy;
    best_feature_percentage_per_sub(sub) = best_feature_percentage;
end

disp('每个子集的最大精度与对应的特征百分比：');
for sub = 1:20
    disp(['Sub ', num2str(sub), ': 最大精度 = ', num2str(max_accuracy_per_sub(sub)), ', 最佳特征百分比 = ', num2str(best_feature_percentage_per_sub(sub))]);
end

    
