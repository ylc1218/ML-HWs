function [best_log_lambda, best_e_train, best_e_val, best_e_out] = hw4_q16()
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
    best_e_train = 1;
    best_g = zeros(3,1);
    
    vec_etrain=zeros(13,1);
    t=1;
    for i=-10:2
        lambda = 10^i;
        g = inv(d_train_x_app.'*d_train_x_app+lambda*eye(D+1))*d_train_x_app.'*d_train_y;
        e_train = cal_err(d_train_x_app, d_train_y, g);
        
        if e_train<=best_e_train
            best_log_lambda = i;            
            best_e_train = e_train;
            best_g = g;
        end            
        vec_etrain(t)=e_train;
        t=t+1;
    end
        
    best_e_val = cal_err(d_val_x_app, d_val_y, best_g);
    best_e_out = cal_err(test_x_app, test_y, best_g);
    
    xmarkers = -10:2; % place markers at these x-values
    ymarkers = vec_etrain;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
        
    title('Hw4-q16');
    xlabel('log_{10}\lambda');
    ylabel('E_{train}');
    
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