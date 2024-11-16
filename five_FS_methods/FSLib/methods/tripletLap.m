function L = tripletLap(X, k)
    % 计算数据点之间的距离
    D = pdist2(X', X');
    
    % 找到每个数据点的 k 个最近邻
    [~, indices] = mink(D, k+1, 2);
    
    % 创建邻接矩阵
    A = zeros(size(X, 2));
    for i = 1:size(X, 2)
        A(i, indices(i, 2:end)) = 1; % 除了自身外，将最近邻的点连接起来
    end
    
    % 计算度矩阵
    D = diag(sum(A, 2));
    
    % 计算拉普拉斯矩阵
    L = D - A;
end
