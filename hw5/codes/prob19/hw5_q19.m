function eout_arr=hw5_q19()
    digit=0;
    
    [train_x, train_y] = train_input();
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    [test_x, test_y] = test_input();
    test_y(test_y~=digit)=-1;
    test_y(test_y==digit)=1;
    
    arr_log_gamma=[0,1,2,3,4];
    cnt=size(arr_log_gamma,2);
    
    eout_arr=zeros(1,cnt);
    c=0.1;
    for i=1:cnt
        gamma=10^arr_log_gamma(i);
        para = sprintf('-c %g -t 2 -g %g', c,gamma);
        model = svmtrain(train_y,train_x, para);
        [~, accuracy, ~] = svmpredict(test_y, test_x, model');
        eout_arr(i)=(100-accuracy(1))/100;
    end
    
    xmarkers = arr_log_gamma; % place markers at these x-values
    ymarkers = eout_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
    title('Hw5-q19');
    xlabel('log_{10}\gamma');
    ylabel('E_{out}');


end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end

function [test_x, test_y] = test_input()
    test_x = textread('features.test');
    test_y = test_x(:, 1);
    test_x = test_x(:, 2:end);
end