function arr_sum=hw5_q17()
    [train_x, train_y] = train_input();
    digit = 8;
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    arr_log_c=[-6,-4,-2,0,2];
    cnt=size(arr_log_c,2);
    
    arr_sum=zeros(1,cnt);
    for i=1:cnt
        para = sprintf('-c %g -t 1 -d 2 -g 1 -r 1',10^arr_log_c(i));
        model = svmtrain(train_y,train_x,para);
        alpha=abs(model.sv_coef);
        arr_sum(i)=sum(alpha);
    end
    
    xmarkers = arr_log_c; % place markers at these x-values
    ymarkers = arr_sum;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
    %strValues = strtrim(cellstr(num2str([xmarkers(:) ymarkers(:)],'(%d,%d)')));
    %text(xmarkers,ymarkers, strValues,'Color','red','VerticalAlignment','bottom');
    
    title('Hw5-q17');
    xlabel('log_{10}c');
    ylabel('\Sigma_{n=1}^N \alpha_n');
end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end
