function [w, eout] = hw3_q18()
    init();
    
    [train_x, train_y] = train_input();
    [N_train, D] =  size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    T=2000;
    ETA = 0.001; %0.001 for Q18, 0.01 for Q19
    w=zeros(D+1,1);
    for t=1:T
        g_ein = cal_gradient_ein(train_x_app, train_y, w);
        w = w-ETA*g_ein;
    end
    
    [test_x, test_y] = test_input();
    [N_test, ~] = size(test_x);
    eout = cal_err([ones(N_test, 1) test_x], test_y, w);    
end

function err = cal_err(x, y, w)
    [N,~] = size(x);
    myY = sign(x*w);
    myY(myY==0) = -1; % take sign(0) = -1
    diff = myY.*y; %if(same)->1, else->-1
    err = sum(diff~=1)/N;
end

function g_ein = cal_gradient_ein(x, y, w)
    [N,D] = size(x);
    sum=zeros(D, 1);
    for n=1:N
        xn = x(n,:); % size(xn) = (1, D); the n-th row
        yn = y(n);
        sum = sum + (1/(1+exp(yn*xn*w))*(-yn*xn)).' ; 
    end
    g_ein = sum/N;
end

function init()
    clc
    clear
end


function [test_x, test_y] = test_input()        
    test_x = textread('hw3_test.dat');
    test_y = test_x(:, end);
    test_x = test_x(:, 1:end-1);    
end

function [train_x, train_y] = train_input()
    train_x = textread('hw3_train.dat');
    train_y = train_x(:, end);
    train_x = train_x(:, 1:end-1);
end