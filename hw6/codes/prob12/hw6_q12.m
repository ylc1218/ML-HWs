function [ein_1,alpha_1]=hw6_q12()
    train_data = textread('hw2_adaboost_train.dat');
    train_x = train_data(:, 1:end-1);
    train_y = train_data(:, end);
    [N,D]=size(train_x);
    T=300;
    
    u(1:N)=1/N;
    ein_g_arr=zeros(1,T);
    for t=1:T %iteration
        [best_ein,best_u]=deal(1,u);
        for d=1:D %for each dimension
            sorted_train_x=sortrows(train_x, d);
            for n = 1:N %for each midpoints
                if (n==1)
                    theta=sorted_train_x(1,d)-1; %theta=-INF
                else
                    theta=(sorted_train_x(n,d)+sorted_train_x(n-1,d))/2;
                end
                
                for s=[1,-1] %s=[+1,-1]
                    [tmp_u, ein] = cal_error(train_x(:,d), train_y, theta, s, u);
                    if(ein<best_ein)
                    	[best_ein,best_u]=deal(ein,tmp_u);
                        [g.d, g.theta, g.s] = deal(d, theta, s);
                    end
                end
            end
        end
        [~,ein_g_arr(t)]=cal_error(train_x(:,g.d),train_y, g.theta, g.s, ones(1,N)./N);
        u=best_u; %update u
    end
    ein_1=ein_g_arr(1);
    alpha_1=log(sqrt((1-ein_1)/ein_1));

    %plot
    xmarkers = 1:t; % place markers at these x-values
    ymarkers = ein_g_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw6-q12');
    xlabel('t');
    ylabel('E_{in}(g_t)');
end

function [u,err] = cal_error(in, out, theta, s, u)
    myout = s*sign(in-theta);
    myout(myout==0) = s; % take sign(0) = s
    
    err = u*(myout~=out);  % u*[[myout !=out]]
    
    %cal new u
    n_usum=sum(u(myout~=out));
    eps=n_usum/sum(u);
    t=sqrt((1-eps)/eps); %scaling factor
    u(myout==out)=u(myout==out)/t;
    u(myout~=out)=u(myout~=out)*t;
end