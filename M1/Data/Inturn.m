


load G4_PeakSfLocListTotal.mat
cluster_idx = [];
G4_RespAvg = [];
for i = 1:12
    idx = find(G4_PeakSfLocListTotal(:,13) == i);
    cluster_idx = cat(1,cluster_idx,idx);
    G4_RespAvg = cat(2,G4_RespAvg,G4_RspAvg(:,idx));
end

save G4_RespAvg.mat G4_RespAvg