function arr_norm_w=hw5_q15()
    digit=0;
    
    [train_x, train_y] = train_input();
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    arr_log_c=[-6,-4,-2,0,2];
    cnt=size(arr_log_c,2);
    
    arr_norm_w=zeros(1,cnt);
    for i=1:cnt
        c=10^arr_log_c(i);
        para = sprintf('-c %g -t 0', c);
        model = svmtrain(train_y,train_x,para); 
        w = model.SVs' * model.sv_coef;
        arr_norm_w(i) = norm(w);
    end
    
    
    xmarkers = arr_log_c; % place markers at these x-values
    ymarkers = arr_norm_w;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
      
    %plot(arr_log_c,arr_norm_w);
    title('Hw5-q15');
    xlabel('log_{10}c');
    ylabel('||w||');
end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end