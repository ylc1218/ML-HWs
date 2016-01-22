function ein_arr=hw8_q12()
    [train_x,train_y]=input('hw8_train.dat');
    [N,~]=size(train_x);
    distM=pdist2(train_x,train_x);
    
    K=[1,3,5,7,9];
    
    ein_arr=zeros(1,size(K,2));
    for kid=1:size(K,2)
        predict=zeros(N,1);
        k=K(kid);
        for i=1:N
            [~,idx]=sort(distM(i,:));
            predict(i)=mode(train_y(idx(1:k)));
        end
        ein_arr(kid)=cal_err(predict,train_y);
    end
    %plot
    xmarkers = K; % place markers at these x-values
    ymarkers = ein_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2);
      
    title('Hw8-q12');
    xlabel('K');
    ylabel('E_{in}(g_{k-nbor})');
end

function [x,y]=input(fname)
    data = textread(fname);
    x=data(:,1:end-1);
    y=data(:,end);
end

function err=cal_err(myout, out)
    N=size(out,1);
    err = sum(myout~=out)/N;
end