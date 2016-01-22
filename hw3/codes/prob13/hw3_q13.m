function avg_ein = hw3_q13()
    init();
    global N;
    T = 1000;
    vec_ein = zeros(1, T);
 
    for i=1:T
        [train_x, train_y] = gen_testcase();
        Wlin = pinv([ones(N, 1) train_x])*train_y; %pinv : pseudo inverse
        vec_ein(i) = vec_ein(i) + cal_err([ones(N, 1) train_x], train_y, Wlin);
    end
    avg_ein = sum(vec_ein)/T;
    
    %histogram
    hist(vec_ein);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    
    title('Histogram of Hw3-q13');
    xlabel('Error rate');
    ylabel('Frequency');
end

function init()
    clc
    clear
    global N;
    N=1000;
end

function err = cal_err(x, y, w)
    global N;
    myY = sign(x*w);
    myY(myY==0) = -1; % % take sign(0) = -1
    diff = myY.*y;
    err = sum(diff~=1)/N;
end

function [train_x, train_y] = gen_testcase()
    global N;
    minval = -1;
    maxval = 1;    
    err_rate = 0.1;
    
    in = (maxval-minval).*rand(N,2) + minval; % gen N pairs
    err_id = randperm(N,N*err_rate); % N個數裡面取 N*err_rate個出來 (要flip y的)
    out = sign(in(:,1).*in(:,1)+in(:,2).*in(:,2)-0.6); %y  = sign(x1^2 + x2^2 -0.6)
    out(err_id) = out(err_id)*-1;
    train_x = in;
    train_y = out;
end