function ein_arr=hw8_q16()
    [train_x,train_y]=input('hw8_train.dat');
    [N,~]=size(train_x);
    G=[0.001,0.1,1,10,100];
    nG=size(G,2);
    
    ein_arr=zeros(1,nG);
    for gid=1:nG
        distM=zeros(N,N);
        gamma=G(gid);
        predict=zeros(N,1);
        for i=1:N
            for j=i:N
                d=train_x(i,:)-train_x(j,:);
                dist=exp(-gamma*(d*d.'));
                [distM(i,j), distM(j,i)]=deal(dist,dist);
            end
            predict(i)=sign(distM(i,:)*train_y);
        end
        ein_arr(gid)=cal_err(predict, train_y);
    end
    %plot
    xmarkers = G; % place markers at these x-values
    ymarkers = ein_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2);
      
    title('Hw8-q16');
    xlabel('gamma');
    ylabel('E_{in}(g_{uniform})');

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