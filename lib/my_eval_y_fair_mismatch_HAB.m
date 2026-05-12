function [res]= my_eval_y_fair_mismatch_HAB(Y, g, predY)
%  This code is used for the case when the cluster number in the predict
%  label is not the number of true cluster
%
if size(Y,2) ~= 1
    Y = Y';
end
if size(predY,2) ~= 1
    predY = predY';
end

G = full(ind2vec(g'))';
predY2 = full(ind2vec(predY'))';
C = G'*predY2;
fair = compute_fair(C);
mnce = MNCE(C);
fb = compute_f_b(G, predY2);


nSmp = length(Y);
uY = unique(Y);
nclass = length(uY);
Y0 = zeros(nSmp,1);
if nclass ~= max(Y)
    for i = 1:nclass
        Y0(Y == uY(i)) = i;
    end
    Y = Y0;
end

uY = unique(predY);
nclass = length(uY);
predY0 = zeros(nSmp,1);
if nclass ~= max(predY)
    for i = 1:nclass
        predY0(predY == uY(i)) = i;
    end
    predY = predY0;
end


[newIndx] = bestMap_v2(Y,predY);
acc = mean(Y==newIndx);
nmi = mutual_info(Y,newIndx);
purity = pur_fun(Y,newIndx);
[AR,RI,MI,HI] = RandIndex(Y, newIndx);
% [fscore,precision,recall] = compute_f(Y, newIndx);
fscore = 0;
precision = 0;
recall = 0;
nCluster = length(unique(Y));
    
ys = sum(predY2, 1);
[entropy,bal, SDCS, RME] = BalanceEvl_new(nCluster, ys);

F_acc_entroy = 2 .* acc .* entropy ./ (acc + entropy);
F_acc_RME = 2 .* acc .* RME ./ (acc + RME);
F_nmi_entroy = 2 .* nmi .* entropy ./ (nmi + entropy);
F_nmi_RME = 2 .* nmi .* RME ./ (nmi + RME);

F_acc_fair = 2 .* acc .* fair ./ (acc + fair);
F_acc_mnce = 2 .* acc .* mnce ./ (acc + mnce);
F_nmi_fair = 2 .* nmi .* fair ./ (nmi + fair);
F_nmi_mnce = 2 .* nmi .* mnce ./ (nmi + mnce);

F_acc_fb = 2 .* acc .* fb ./ (acc + fb);
F_nmi_fb = 2 .* nmi .* fb ./ (nmi + fb);

res = [acc, nmi, fair, mnce, fb, entropy, bal, RME, purity, AR, RI, MI, HI, fscore, precision, recall, SDCS, F_acc_entroy,...
    F_acc_RME, F_nmi_entroy, F_nmi_RME, F_acc_fair, F_acc_mnce, F_acc_fb, F_nmi_fair, F_nmi_mnce, F_nmi_fb];