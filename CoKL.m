function [K_x_new, K_y_new] = CoKL(X, Y, a, b, c)
%
% Incomplete data X and Y, each row represents an instance and each column represents an attributes
% c is the number of instances shared by both views.
% a is the number of instances only in view Y.
% b is the number of instance only in view X.
% X(c+1:c+a,:) are the missing data in X and Y(c+a+1:end, :) are the missing data in Y.
% Implemented by Weixiang Shao (wshao4@uic.edu)

    if size(X,1) ~= size(Y,1)
        error('The number of rows must be the same');
        return;
    end
    N = size(X,1);
    d1 = size(X,2);
    d2 = size(X,2);

    % First fill the missing data with average features.
    X(c+1:c+a,:) = repmat((sum(X) - sum(X(c+1:c+a,:)))/(c+b), a,1);
    Y(c+a+1:end,:) = repmat(sum(Y(1:c+a,:))/(c+a), b, 1);

    K_x = X * X';
    K_y = Y * Y';

    D_x = diag(sum(K_x,2));
    D_y = diag(sum(K_y,2));

    L_x = D_x - K_x;
    L_y = D_y - K_y;

    % compute A and B, s.t. K_x = A*A' and K_y = B*B'
    % Here, we can just assign A = Y and B = X, if we use linear kernel.
    A = Y;
    B = X;
    A_c = A(1:c,:);
    A_a = A(c+1:c+a,:);
    A_b = A(c+a+1:c+a+b,:);

    B_c = B(1:c,:);
    B_a = B(c+1:c+a,:);
    B_b = B(c+a+1:c+a+b,:);

    A_b_prev = zeros(size(A(c+a+1:c+a+b,:)));
    B_a_prev = zeros(size(B(c+1:c+a,:)));

    epsilon = 0.02;
    count = 0;
    error_A = zeros(10000,1);
    error_B = zeros(10000,1);



    count_ = 0;
    K_x_prev = zeros(size(K_x));
    K_y_prev = zeros(size(K_y));
    while (max(max(abs(K_x_prev-K_x)))>0.000001 || max(max(abs(K_y_prev-K_y)))>0.000001) && count_<100

        K_x_prev = K_x;
        K_y_prev = K_y;
        K_x_cc = K_x(1:c,1:c);
        K_x_ca = K_x(1:c,c+1:c+a);
        K_x_cb = K_x(1:c,c+a+1:c+a+b);
        K_x_ac = K_x(c+1:c+a, 1:c);
        K_x_aa = K_x(c+1:c+a, c+1:c+a);
        K_x_ab = K_x(c+1:c+a, c+a+1:c+a+b);
        K_x_bc = K_x(c+a+1:c+a+b, 1:c);
        K_x_ba = K_x(c+a+1:c+a+b, c+1:c+a);
        K_x_bb = K_x(c+a+1:c+a+b, c+a+1:c+a+b);


        K_y_cc = K_y(1:c,1:c);
        K_y_ca = K_y(1:c,c+1:c+a);
        K_y_cb = K_y(1:c,c+a+1:c+a+b);
        K_y_ac = K_y(c+1:c+a, 1:c);
        K_y_aa = K_y(c+1:c+a, c+1:c+a);
        K_y_ab = K_y(c+1:c+a, c+a+1:c+a+b);
        K_y_bc = K_y(c+a+1:c+a+b, 1:c);
        K_y_ba = K_y(c+a+1:c+a+b, c+1:c+a);
        K_y_bb = K_y(c+a+1:c+a+b, c+a+1:c+a+b);

        L_x_cc = L_x(1:c,1:c);
        L_x_ca = L_x(1:c,c+1:c+a);
        L_x_cb = L_x(1:c,c+a+1:c+a+b);
        L_x_ac = L_x(c+1:c+a, 1:c);
        L_x_aa = L_x(c+1:c+a, c+1:c+a);
        L_x_ab = L_x(c+1:c+a, c+a+1:c+a+b);
        L_x_bc = L_x(c+a+1:c+a+b, 1:c);
        L_x_ba = L_x(c+a+1:c+a+b, c+1:c+a);
        L_x_bb = L_x(c+a+1:c+a+b, c+a+1:c+a+b);


        L_y_cc = L_y(1:c,1:c);
        L_y_ca = L_y(1:c,c+1:c+a);
        L_y_cb = L_y(1:c,c+a+1:c+a+b);
        L_y_ac = L_y(c+1:c+a, 1:c);
        L_y_aa = L_y(c+1:c+a, c+1:c+a);
        L_y_ab = L_y(c+1:c+a, c+a+1:c+a+b);
        L_y_bc = L_y(c+a+1:c+a+b, 1:c);
        L_y_ba = L_y(c+a+1:c+a+b, c+1:c+a);
        L_y_bb = L_y(c+a+1:c+a+b, c+a+1:c+a+b);

        A_c = A(1:c,:);
        A_a = A(c+1:c+a,:);
        A_b = A(c+a+1:c+a+b,:);

        B_c = B(1:c,:);
        B_a = B(c+1:c+a,:);
        B_b = B(c+a+1:c+a+b,:);

        A_b = -(inv(L_x_bb))*(L_x_cb')*A_c - (inv(L_x_bb))*(L_x_ab')*A_a;
        A_b_prev = A(c+a+1:c+a+b,:);
        A(c+a+1:c+a+b,:) = A_b;

        B_a = -(inv(L_y_aa))*(L_y_ca')*B_c - (inv(L_y_aa))*(L_y_ab)*B_b;
        B_a_prev = B(c+1:c+a,:);
        B(c+1:c+a,:) = B_a;
        count_ = count_ +1;
        error_A(count_) = sum(sum(abs(B_a_prev - B_a)));
        error_B(count_) = sum(sum(abs(A_b_prev-A_b)));


        K_x = B * B';
        K_y = A * A';

        fprintf('K_x diff is %f, K_y diff is %f\n', sum(sum(abs(K_x-K_x_original))), sum(sum(abs(K_y-K_y_original))));
        fprintf('B_x diff is %f, A_y diff is %f\n', error_B(count_), error_A(count_));
        fprintf('B_x diff original %f, A_y diff original %f\n', sum(sum(abs(B-V_1))), sum(sum(abs(A-V_2))));

        D_x = zeros(size(K_x));
        D_y = zeros(size(K_y));

        for i = 1:size(K_x,1)
            D_x(i,i) = sum(K_x(i,:));
        end

        for i = 1:size(K_y,1)
            D_y(i,i) = sum(K_y(i,:));
        end
        % compute L_x, L_y
        L_x = D_x - K_x;
        L_y = D_y - K_y;
    end
    diff(count_missing).x = K_x - V_1_original(:,1:end-1)*V_1_original(:,1:end-1)';
    diff(count_missing).y = K_y - V_2_original(:,1:end-1)*V_2_original(:,1:end-1)';

    fprintf('count_ is %d\n', count_);
    K_x_new = K_x;
    K_y_new = K_y;
end
