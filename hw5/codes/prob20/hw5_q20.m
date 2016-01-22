function cnt_arr=hw5_q20()
    digit=0;
    c=0.1;
    T=100;
    N_VAL=1000;
    [train_x, train_y] = train_input();
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    
    arr_log_gamma=[0,1,2,3,4];
    cnt=size(arr_log_gamma,2);
    cnt_arr=zeros(1,cnt);
    hist_arr=zeros(1,T);
    for t=1:T
        %split data into d_train, d_val
        [d_val_x, d_val_y, d_train_x, d_train_y]=split_train(train_x, train_y, N_VAL);
            
        best_eval=100;
        best_log_gamma=arr_log_gamma(1);
        
        for i=1:cnt
            gamma=10^arr_log_gamma(i);
            para = sprintf('-c %g -t 2 -g %g', c,gamma);
            model = svmtrain(d_train_y,d_train_x, para);
            [~, accuracy, ~] = svmpredict(d_val_y, d_val_x, model');
            eval=(100-accuracy(1))/100;
            
            if(eval<best_eval)
                best_eval=eval;
                best_log_gamma=arr_log_gamma(i);
            end
        end
        hist_arr(t)=best_log_gamma;
        tmp=find(arr_log_gamma==best_log_gamma);
        cnt_arr(tmp)=cnt_arr(tmp)+1;
    end
    
    hist(hist_arr,arr_log_gamma);
    set(gca, 'XTick', arr_log_gamma);
    
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey);
    title('Histogram of Hw5-q20');
    xlabel('log_{10}\gamma');
    ylabel('Number of times selected');


end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end

function [d_val_x, d_val_y, d_train_x, d_train_y] = split_train(train_x, train_y, N_VAL)
    N=size(train_x,1);
    val_id = randperm(N,N_VAL);
    d_val_x = train_x(val_id, :);
    d_val_y = train_y(val_id);

    d_train_x = train_x;
    d_train_y = train_y;
    d_train_x(val_id,:)=[];
    d_train_y(val_id)=[];
end
