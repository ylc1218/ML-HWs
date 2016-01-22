function eout_g_1=hw6_q17()
    train_data = textread('hw2_adaboost_train.dat');
    train_x = train_data(:, 1:end-1);
    train_y = train_data(:, end);
    [N,D]=size(train_x);
    
    [test_x, test_y] = test_input();
    [test_N,~]=size(test_x);

    T=300;
    
    u(1:N)=1/N;
    eout_g_arr=zeros(1,T);
    g(T)=struct('d',[],'theta',[],'s',[]);
    
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
                    [tmp_u, ein] = cal_error(train_x(:,d), train_y, theta, s, u);
                    if(ein<=best_ein)
                    	[best_ein, best_u]=deal(ein, tmp_u);
                        [g(t).d, g(t).theta, g(t).s]=deal(d, theta, s); %update g(t)
                    end
                end
            end
        end
        [~,eout_g_arr(t)]=cal_error(test_x(:,1), test_y, g(t).theta, g(t).s, ones(1,test_N)./test_N);
        u=best_u; %update u
    end
    eout_g_1 = eout_g_arr(1);

    %plot
    xmarkers = 1:t; % place markers at these x-values
    ymarkers = eout_g_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw6-q17');
    xlabel('t');
    ylabel('E_{out}(g_t)');
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

function [u, err] = cal_error(in, out, theta, s, u)
    myout = s*sign(in-theta);
    err = u*(myout~=out);  % u*[[myout !=out]]
    
    %cal new u
    n_usum=sum(u(myout~=out));
    eps=n_usum/sum(u);
    t=sqrt((1-eps)/eps); %scaling factor
    u(myout==out)=u(myout==out)/t;
    u(myout~=out)=u(myout~=out)*t;
end