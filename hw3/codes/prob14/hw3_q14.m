function [avg_w3, Wlin] = hw3_q14()
    init();
    T=1000;
    vec_w3=zeros(T:1);
    for t=1:T
        [train_x, train_y] = gen_testcase();
        transform_x = transform(train_x);
        Wlin = pinv(transform_x)*train_y; %pinv : pseudo inverse
        vec_w3(t)=Wlin(4);
    end
    avg_w3=sum(vec_w3)/T;
    
    %histogram
    hist(vec_w3);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    
    title('Histogram of Hw3-q14');
    xlabel('w3');
    ylabel('Frequency');
    
end

function init()    
    clc
    clear
    global N;
    N=1000;
end

function [transform_x] = transform(x)
    global N;
    transform_x = [ones(N,1) x, x(:, 1).*x(:, 2) x(:,1).*x(:,1) x(:,2).*x(:,2)]; %(1,x1,x2,x1x2,x1x1,x2x2)
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