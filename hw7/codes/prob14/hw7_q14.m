function ein=hw7_q14()
    [train_x,train_y]=train_input();
    root = hw7_buid_tree(train_x, train_y);
    ein=cal_error(root, train_x, train_y)/size(train_x,1);
end

function [train_x,train_y]=train_input()
    train_data = textread('hw7_train.dat');
    train_x=train_data(:,1:end-1);
    train_y=train_data(:,end);
end

function error = cal_error(node, in, out)
    if(node.isLeaf)
        error=sum(out~=node.val);
    else
        stump_out = node.s*sign(in(:,node.d)-node.theta);
        error=cal_error(node.lch, in(stump_out==-1,:), out(stump_out==-1));
        error=error+cal_error(node.rch, in(stump_out==1,:), out(stump_out==1));
    end
end