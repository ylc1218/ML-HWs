function eout_G=hw6_q18()
    train_data = textread('hw2_adaboost_train.dat');
    train_x = train_data(:, 1:end-1);
    train_y = train_data(:, end);
    [N,D]=size(train_x);
    
    [test_x, test_y] = test_input();
    T=300;
    
    u(1:N)=1/N;
    eout_G_arr=zeros(1,T);
    g(T)=struct('d',[],'theta',[],'s',[], 'alpha',[]);
    
    for t=1:T %iteration
        [best_ein,best_u]=deal(1,u);
        
        for d=1:D %for each dimension
            sort_train_x=sortrows(train_x,d);
            for n = 1:N %for each midpoint
                if (n==1)
                    theta=sort_train_x(1,d)-1; %theta=-INF
                else
                    theta=(sort_train_x(n,d)+sort_train_x(n-1,d))/2;
                end
                    
                for s=[1,-1] %s=[+1,-1]
                    [tmp_u, ein, alpha] = cal_error(train_x(:,d), train_y, theta, s, u);
                    if(ein<=best_ein)
                    	[best_ein, best_u]=deal(ein, tmp_u);
                        [g(t).d, g(t).theta, g(t).s, g(t).alpha]=deal(d, theta, s, alpha); %update g(t)
                    end
                end
            end
        end
        u=best_u; %update u
        eout_G_arr(t)=cal_G_error(test_x,test_y, g(1:t));
    end
    eout_G = eout_G_arr(T);

    %plot
    xmarkers = 1:t; % place markers at these x-values
    ymarkers = eout_G_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw6-q18');
    xlabel('t');
    ylabel('E_{out}(G_t)');
end

function [test_x, test_y] = test_input()
    test_data = textread('hw2_adaboost_test.dat');
    test_y = test_data(:, end);
    test_x = test_data(:, 1:end-1);
end

function [train_x, train_y] = sort_input(train_data, d)
    sortrows(train_data,d);
    train_y = train_data(:, end);
    train_x = train_data(:, 1:end-1);
end

function err=cal_G_error(in, out, g)
    T=size(g,2);
    N=size(out,1);
    sum_out=zeros(N,1);
    for t=1:T
       gout = g(t).s*sign(in(:,g(t).d)-g(t).theta);
       sum_out=sum_out+g(t).alpha*gout;
    end
    err=sum(out~=sign(sum_out))/N;
end

function [u, err, alpha] = cal_error(in, out, theta, s, u)
    myout = s*sign(in-theta);    
    err = u*(myout~=out);  % u*[[myout !=out]]
    
    %cal new u
    n_usum=sum(u(myout~=out));
    eps=n_usum/sum(u);
    t=sqrt((1-eps)/eps); %scaling factor
    u(myout==out)=u(myout==out)/t;
    u(myout~=out)=u(myout~=out)*t;
    
    %cal new alpha
    alpha=log(t);
end