function  [best_log_lambda, best_e_cv] = hw4_q19()
    init();
    global F, global F_size, global D;
        
    [train_x, train_y] = train_input();        
    [N_train, D]= size(train_x);
    train_x_app = [ones(N_train, 1) train_x];    
        
    F_size = N_train/F;        
    best_e_cv = F;
    best_log_lambda = 2;
    
    vec_ecv=zeros(13,1);
    t=1;
    
    %for each lambda
    for i=-10:2       
        e_cv = cal_e_cv(train_x_app, train_y, 10^i); %calculate e_cv for lambda=10^i
                
        if e_cv<=best_e_cv
            best_e_cv = e_cv;
            best_log_lambda = i;
        end
        vec_ecv(t)=e_cv;
        t=t+1;
    end
    
    xmarkers = -10:2; % place markers at these x-values
    ymarkers = vec_ecv;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
        
    title('Hw4-q19');
    xlabel('log_{10}\lambda');
    ylabel('E_{cv}');
   
    
          
end

function init()
    clc
    clear
    
    global F;
    F=5;  % # folds    
end

function e_cv = cal_e_cv(train_x_app, train_y, lambda)
    global F, global F_size, global D;
    
    e_cv_sum = 0;
    for f=1:F            
        val_row_id = [(f-1)*F_size+1:f*F_size]; % id of validation rows
        d_val_x_app = train_x_app(val_row_id, :);
        d_val_y = train_y(val_row_id);

        d_train_x_app = train_x_app;
        d_train_x_app(val_row_id, :) = []; % trim validation rows
        d_train_y = train_y;
        d_train_y(val_row_id)=[]; %trim validation cols

        g = inv(d_train_x_app.'*d_train_x_app+lambda*eye(D+1))*d_train_x_app.'*d_train_y;
        e_val = cal_err(d_val_x_app, d_val_y, g);
        e_cv_sum = e_cv_sum + e_val;
    end
    e_cv = e_cv_sum/F;
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