function [best_log_lambda, e_in, e_out] = hw4_q18()
    init();
    [test_x, test_y] = test_input();
    [train_x, train_y] = train_input();
    
    [N_test, D] = size(test_x);
    [N_train, ~]= size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    test_x_app = [ones(N_test, 1) test_x];
    
    %split data into d_train, d_val
    d_train_x_app = train_x_app(1:120, :);
    d_train_y = train_y(1:120);
    
    d_val_x_app = train_x_app(121:end, :);
    d_val_y = train_y(121:end);
        
    best_log_lambda = 2;
    best_e_val = 1;    
    
    for i=-10:2
        lambda = 10^i;
        g = inv(d_train_x_app.'*d_train_x_app+lambda*eye(D+1))*d_train_x_app.'*d_train_y;
        e_val = cal_err(d_val_x_app, d_val_y, g);
        
        
        if e_val<=best_e_val
            best_log_lambda = i;            
            best_e_val = e_val;
        end            
    end
    
    g = inv(train_x_app.'*train_x_app+(10^best_log_lambda)*eye(D+1))*train_x_app.'*train_y;
    e_in = cal_err(train_x_app, train_y, g);    
    e_out = cal_err(test_x_app, test_y, g);
    
end

function init()
    clc
    clear
end

function err = cal_err(x, y, w)
    [N,~] = size(x);
    myY = sign(x*w);
    myY(myY==0)=-1;
    err = sum(myY~=y)/N;
end

function [test_x, test_y] = test_input()        
    test_x = textread('hw4_test.dat');
    test_y = test_x(:, end);
    test_x = test_x(:, 1:end-1);    
end

function [train_x, train_y] = train_input()
    train_x = textread('hw4_train.dat');
    train_y = train_x(:, end);
    train_x = train_x(:, 1:end-1);
end