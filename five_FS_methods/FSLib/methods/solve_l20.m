function X_sparse = solve_l20(X, k)
    % 迭代算法来解决 L2,0 范数正则化问题
    % X: 待稀疏化的矩阵
    % k: 每列最多允许非零元素的个数
    
    % 初始化稀疏矩阵
    X_sparse = X;
    
    % 迭代次数
    maxIter = 100;
    
    % 迭代过程
    for iter = 1:maxIter
        % 计算每一列的 L2 范数
        norms = sqrt(sum(X_sparse.^2, 1));
        
        % 找到每列的排序索引
        [~, sorted_indices] = sort(norms, 'descend');
        
        % 对于每列，保留前 k 个非零元素，其他置零
        for col = 1:size(X_sparse, 2)
            col_indices = sorted_indices(:, col); % 使用圆括号来索引
            X_sparse(col_indices(k+1:end), col) = 0;
        end
    end
end




