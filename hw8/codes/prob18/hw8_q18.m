function eout_arr=hw8_q18()
    [train_x,train_y]=input('hw8_train.dat');
    [test_x,test_y]=input('hw8_test.dat');
    
    [train_N,~]=size(train_x);
    [test_N,~]=size(test_x);
    
    G=[0.001,0.1,1,10,100];
    nG=size(G,2);
    
    eout_arr=zeros(1,nG);
    for gid=1:nG
        distM=zeros(test_N,train_N);
        gamma=G(gid);
        predict=zeros(test_N,1);
        for i=1:test_N
            for j=1:train_N
                d=test_x(i,:)-train_x(j,:);
                dist=exp(-gamma*(d*d.'));
                distM(i,j)=dist;
            end
            predict(i)=sign(distM(i,:)*train_y);
        end
        eout_arr(gid)=cal_err(predict, test_y);
    end
    %plot
    xmarkers = G; % place markers at these x-values
    ymarkers = eout_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2);
      
    title('Hw8-q18');
    xlabel('gamma');
    ylabel('E_{out}(g_{uniform})');
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