function [u_sum_2,u_sum_T]=hw6_q15()
    train_data = textread('hw2_adaboost_train.dat');
    train_x = train_data(:, 1:end-1);
    train_y = train_data(:, end);
    [N,D]=size(train_x);
    T=300;
    u(1:N)=1/N;
    
    u_sum_arr=zeros(1,T);
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
                    end
                end
            end
        end
        u_sum_arr(t)=sum(u);
        u=best_u; %update u
    end
    u_sum_2=u_sum_arr(2);
    u_sum_T=u_sum_arr(T);

    %plot
    xmarkers = 1:t; % place markers at these x-values
    ymarkers = u_sum_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw6-q15');
    xlabel('t');
    ylabel('U_t');
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