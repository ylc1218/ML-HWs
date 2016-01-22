function arr_ein=hw5_q16()
    [train_x, train_y] = train_input();
    digit = 8;
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    arr_log_c=[-6,-4,-2,0,2];
    cnt=size(arr_log_c,2);
    arr_ein=zeros(1,cnt);
    for i=1:cnt
        para = sprintf('-c %g -t 1 -d 2 -g 1 -r 1',10^arr_log_c(i));
        model = svmtrain(train_y,train_x,para);
        %{
            sv_idx=model.sv_indices;
            alpha_y=model.sv_coef;
            X=train_x(sv_idx,:).';
            w=X*alpha_y;
        %}
        [predict_label, accuracy, prob_values] = svmpredict(train_y, train_x, model');
        arr_ein(i)=(100-accuracy(1))/100;
    end
    
    xmarkers = arr_log_c; % place markers at these x-values
    ymarkers = arr_ein;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
    ylim([0,0.1]);
    %plot(arr_log_c,arr_norm_w);
    title('Hw5-q16');
    xlabel('log_{10}c');
    ylabel('E_{in}');
end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end
