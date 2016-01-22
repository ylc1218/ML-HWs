function [ein, eout] = hw4_q13()
    init();
    [test_x, test_y] = test_input();
    [train_x, train_y] = train_input();
    
    [N_test, D] = size(test_x);
    [N_train, ~]= size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    test_x_app = [ones(N_test, 1) test_x];
    
    lambda = 11.26;
    w_reg = inv(train_x_app.'*train_x_app+lambda*eye(D+1))*train_x_app.'*train_y;
    ein = cal_err(train_x_app, train_y, w_reg);
    eout = cal_err(test_x_app, test_y, w_reg);    
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