function eout_arr=hw8_q14()
    [train_x,train_y]=input('hw8_train.dat');
    [test_x,test_y]=input('hw8_test.dat');
    
    [test_N,~]=size(test_x);
    
    distM=pdist2(test_x,train_x);
    
    K=[1,3,5,7,9];
    
    eout_arr=zeros(1,size(K,2));
    for kid=1:size(K,2)
        predict=zeros(test_N,1);
        k=K(kid);
        for i=1:test_N
            [~,idx]=sort(distM(i,:));
            predict(i)=mode(train_y(idx(1:k)));
        end
        eout_arr(kid)=cal_err(predict,test_y);
    end
    %plot
    xmarkers = K; % place markers at these x-values
    ymarkers = eout_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2);
      
    title('Hw8-q14');
    xlabel('K');
    ylabel('E_{out}(g_{k-nbor})');
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