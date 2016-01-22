function ein_arr=hw8_q20()
    data = textread('hw8_nolabel_train.dat');
    K=[2,4,6,8,10];
    nK=size(K,2);
    T=10;
    
    ein_arr=zeros(1,nK);
    for kid=1:nK
        k=K(kid);
        err=0;
        for t=1:T
            [group,mu]=kmeans(data, k);
            err=err+cal_err(data, group, mu);
        end
        ein_arr(kid)=err/T;
    end
    %plot
    xmarkers = K; % place markers at these x-values
    ymarkers = ein_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2);
      
    title('Hw8-q20');
    xlabel('K');
    ylabel('E_{in}');
end

function err=cal_err(data, group, mu)
    K=size(mu,1);
    err=0;
    for k=1:K
        d=data(group==k,:)-repmat(mu(k,:),sum(group==k),1);
        err=err+sum(sum(d.^2));   
    end
    err=err/size(data,1);
end

function [group,mu]=kmeans(data,K)
    N=size(data,1);
    gidx=randperm(N,K);
    mu=data(gidx,:);
    
    %cluster
    group=zeros(1,N);
    n_group=zeros(1,N);
    while(1)
        distM=pdist2(data,mu);
        for i=1:N
            [~,idx]=min(distM(i,:));
            n_group(i)=idx;
        end
        
        if(sum(n_group==group)==N)
            break;
        end
        
        for k=1:K
            mu(k,:)=mean(data(n_group==k,:));
        end
        group=n_group;
    end
end