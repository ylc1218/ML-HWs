function [best_log_lambda, best_ein, best_eout] = hw4_q15()
    init();
    [test_x, test_y] = test_input();
    [train_x, train_y] = train_input();
    
    [N_test, D] = size(test_x);
    [N_train, ~]= size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    test_x_app = [ones(N_test, 1) test_x];
    
    best_log_lambda = 2;
    best_ein = 1;
    best_eout = 1;
    vec_eout=zeros(13,1);
    t=1;
    
    for i=-10:2
        lambda = 10^i;
        w_reg = inv(train_x_app.'*train_x_app+lambda*eye(D+1))*train_x_app.'*train_y;
        eout = cal_err(test_x_app, test_y, w_reg);
        
        if eout<=best_eout
            best_log_lambda = i;
            best_ein = cal_err(train_x_app, train_y, w_reg);
            best_eout = eout;
        end
        vec_eout(t)=eout;
        t=t+1;
    end
    
    xmarkers = -10:2; % place markers at these x-values
    ymarkers = vec_eout;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
        
    title('Hw4-q15');
    xlabel('log_{10}\lambda');
    ylabel('E_{out}');
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