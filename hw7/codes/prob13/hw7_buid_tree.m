function root =hw7_buid_tree(train_x, train_y)
    [root.train_x, root.train_y]=deal(train_x, train_y);
    root = recursive_split(root);
end

function node = recursive_split(node)
    [train_x, train_y]=deal(node.train_x, node.train_y);
    [N,D]=size(train_x);
    best_score=inf;
    
    for d=1:D %for each dimension
        sort_train_x=sortrows(train_x,d); %sort to compute theta

        for n = 1:N %for each midpoint
            if (n==1), theta=-inf; %theta=-INF
            else
                if (sort_train_x(n,d)==sort_train_x(n-1,d))
                    continue;
                else theta=(sort_train_x(n,d)+sort_train_x(n-1,d))/2;
                end
            end
            
            for s=[1,-1] %s=[+1,-1]
                score = cal_stump_score(train_x(:,d), train_y, theta, s); 
                if(score<best_score)
                    best_score=score;
                    [node.d, node.theta, node.s]=deal(d, theta, s); %update node
                end
            end
        end
    end
    
    if(node.theta==-inf) %create leaf : Xn all the same(no stump)
        [node.cnt, node.val, node.isLeaf] = deal(1, mode(node.train_y), true); % leaf's val is most frequent value in train_y 
    else % not leaf, continue recursive
        node.isLeaf=false;
        [lch, rch]=split(node);
        [node.lch, node.rch] = deal(recursive_split(lch), recursive_split(rch)); % recursive
        node.cnt=1+node.lch.cnt+node.rch.cnt;
    end
end

function [lch,rch]=split(node)
    stump_out = node.s*sign(node.train_x(:,node.d)-node.theta);
    [lch.train_x,lch.train_y]=deal(node.train_x(stump_out==-1,:), node.train_y(stump_out==-1));
    [rch.train_x,rch.train_y]=deal(node.train_x(stump_out==+1,:), node.train_y(stump_out==+1));
end

function score=cal_stump_score(in, out, theta, s)
    stump_out = s*sign(in-theta);
    score=0;
    for c=[1,-1]
        cnt=sum(stump_out==c); % size of Dc
        impurity=cal_gini(out(stump_out==c)); %impurity
        score=score+cnt*impurity;
    end
end

function impurity=cal_gini(out)
    N=size(out,1);
    if (N==0)
        impurity=0;
        return;
    end
    
    pure=0;
    for k=[1,-1]
        cnt=sum(out==k); % yn==k
        pure=pure+(cnt/N)^2;
    end
    impurity=1-pure;
end