function min_eps=hw6_q16()
    train_data = textread('hw2_adaboost_train.dat');
    train_x = train_data(:, 1:end-1);
    train_y = train_data(:, end);
    [N,D]=size(train_x);
    
    T=300;
    u(1:N)=1/N;
    
    eps_arr=zeros(1,T);
    
    for t=1:T %iteration
        [best_ein, best_u]=deal(1,u);
        
        for d=1:D %for each dimension
            sort_train_x=sortrows(train_x,d);
            for n = 1:N %for each midpoint
                if (n==1)
                    theta=sort_train_x(1,d)-1; %theta=-INF
                else
                    theta=(sort_train_x(n,d)+sort_train_x(n-1,d))/2;
                end
                    
                for s=[1,-1] %s=[+1,-1]
                    [tmp_u, ein, eps] = cal_error(train_x(:,d), train_y, theta, s, u);
                    if(ein<=best_ein)
                    	[best_ein, best_u, eps_arr(t)]=deal(ein, tmp_u, eps);
                    end
                end
            end
        end
        u=best_u; %update u
    end
    min_eps=min(eps_arr);
    
    %plot
    xmarkers = 1:t; % place markers at these x-values
    ymarkers = eps_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw6-q16');
    xlabel('t');
    ylabel('\epsilon_t');
end

function [u, err, eps] = cal_error(in, out, theta, s, u)
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