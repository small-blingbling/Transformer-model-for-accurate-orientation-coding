%% Generate New Response List by using maximum value between 13-16 and 14-17
%% Usining the temporal data with 24 frames
%% Edited by Guan 20170921
% Line 206 182 299 has changed, be careful

 function A1_GenerateResponseList
%     try
%         load temporal_tuning_Modify.mat;
%     catch
%         A0_NormalizeTT;
%         load temporal_tuning_Modify.mat;
%     end
% 
%     % temporal_tuning
%     % [frames, sf, ori, size, neuron, repeat(1st half of +1 direction, second half of -1 direction)]
% 
%     FrameNum = size(temporal_tuning,1);
%     SfNum = size(temporal_tuning,2);
%     OriNum = size(temporal_tuning,3);
%     SzNum = size(temporal_tuning,4);
%     CellNum = size(temporal_tuning,5);
%     RepeatNum = size(temporal_tuning,6);
% 
%     G4_RspAvgOFFListTotal = zeros(SfNum*SzNum,CellNum,OriNum,RepeatNum);
%     RawData_BadCell = [];
% 
%     for ci=1:CellNum
%         for a = 1:OriNum
%             for b = 1:SfNum
%                 for c = 1:SzNum
%                     OffImage = mean(mean(squeeze(temporal_tuning(5:8,b,a,c,ci,:))));%%%% Average 10 trials 5:8 frames as OFFimage for each condition
%                     for d = 1:RepeatNum
%                         OnImage_two = [mean(squeeze(temporal_tuning(10:12,b,a,c,ci,d))),mean(squeeze(temporal_tuning(11:13,b,a,c,ci,d)))];
%                         OnImage = max(OnImage_two); %%%% Pick the maximum response between 13:16 frames and 14:17 frames 
% %                         OnImage = mean(squeeze(temporal_tuning(14:17,b,a,c,ci,d)));
%                         rsp(d) = (OnImage-OffImage)/OffImage; 
% %                         rsp(find(isnan(rsp)==1))= 0.00001;  
%                     end
%                     G4_RspAvgOFFListTotal(SzNum*(b-1)+c,ci,a,:) = rsp; % 3(sz)* 6(Sf)
%                 end 
%             end
%         end
%     end
% 
%     save G4_RspAvgOFFListTotal.mat G4_RspAvgOFFListTotal; % [Sf*Size, neuron,ori, :]
%     save RawData_BadCell.mat RawData_BadCell;


    %% Pick the maximum sf and orientation
    %% Save the mean response of each condition by avg repeat trials
    load G4_RspAvgOFFListTotal.mat;
    SfNum = 6;
    OriNum = 12;
    SzNum = 3;
    CellNum = size(G4_RspAvgOFFListTotal,2);
    RepeatNum = size(G4_RspAvgOFFListTotal,4);


    SfSzNum = size(G4_RspAvgOFFListTotal,1);
    cellNum = size(G4_RspAvgOFFListTotal,2);
    oln = size(G4_RspAvgOFFListTotal,3);
    trialNum = size(G4_RspAvgOFFListTotal,4);
    G4_RspMeanTrialStdSeListTotal = zeros(SfSzNum,cellNum,oln,4);
    %1.Useful trial avg(<3sd); 2.Useful trial number; 3.Useful trial std; 4.Std/sqrt(trial number);

    for ci = 1:cellNum  %%CellNum
        for ss = 1:SfSzNum
            for j3 = 1:oln %%Orientaion
                rspTemporal = squeeze(G4_RspAvgOFFListTotal(ss,ci,j3,:));
                rspMean = mean(rspTemporal);
                rspStd = std(rspTemporal);
                index = find(abs(rspTemporal-rspMean)>3*rspStd);
                rspTemporal(index) = 0;     %%%%Dlete >3std trials
                rspTrial = length(find(rspTemporal~=0));
                rspMean = mean(rspTemporal(find(rspTemporal~=0)));
                rspStd = std(rspTemporal(find(rspTemporal~=0)));
                rspSe = rspStd./sqrt(rspTrial);
%                 rspMean(find(rspMean==inf))= 0.00001;
%                 rspSe(find(rspMean==inf))= max(rspSe);
%                 
%                 if length(find(isnan(rsp)==1))
%                     RawData_BadCell = [RawData_BadCell,ci];
%                 end
                
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,1) = rspMean;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,2) = rspTrial;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,3) = rspStd;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,4) = rspSe;
            end
        end
    end

    save G4_RspMeanTrialStdSeListTotal.mat G4_RspMeanTrialStdSeListTotal; % [Sf*Size, neuron,ori, mean-trialNum-std-se]


    %% Find the Maximum Response in the matrix
    %%%Pick the maximum response Size in each Sf*Ori condition
    SzNum = 3;
    G4_SfOri_MeanSeListTotal = zeros(SfSzNum/SzNum,cellNum,oln,2);
    %1.Mean; 2.Se
    G4_SzLocListTotal = zeros(SfSzNum/SzNum,cellNum,oln);


    for ci = 1:cellNum
        SfSzNum_Oln_Mean = squeeze(G4_RspMeanTrialStdSeListTotal(:,ci,:,1));
        SfSzNum_Oln_Se = squeeze(G4_RspMeanTrialStdSeListTotal(:,ci,:,4));
            for j4 = 1:SfSzNum/SzNum % for each sf
                New_Mat = SfSzNum_Oln_Mean(j4*SzNum-SzNum+1:j4*SzNum,:); % [sz of current sf, ori]
                New_Mat_Se = SfSzNum_Oln_Se(j4*SzNum-SzNum+1:j4*SzNum,:);
                x = find(New_Mat==max(max(New_Mat)));
                if isempty(x) 
                   RawData_BadCell = [RawData_BadCell,ci];
                   x=1;
                   SzLoc =1;OriLoc=1;
                else
                    [SzLoc, OriLoc] = find(New_Mat==max(max(New_Mat))); % max index of size and max ori
                end
%                  [SzLoc,OriLoc] = ind2sub([SzNum,oln],x);            
                G4_SfOri_MeanSeListTotal(j4,ci,:,1) = squeeze(New_Mat(SzLoc(1),:)); % [sf, neuron, ori, avg se]
                G4_SfOri_MeanSeListTotal(j4,ci,:,2) = squeeze(New_Mat_Se(SzLoc(1),:));
                G4_SzLocListTotal(j4,ci,:) = j4*SzNum-SzNum+SzLoc(1); % max size index for current af and ori. % [sf, neuron, ori]
            end
    end


    save G4_SfOri_MeanSeListTotal.mat G4_SfOri_MeanSeListTotal;
    save G4_SzLocListTotal.mat G4_SzLocListTotal;

    %% Pick the maximum Ori&Sf response

    load G4_SfOri_MeanSeListTotal.mat;
    load G4_SzLocListTotal.mat

    G4_RawDataforFITTINGListTotal = zeros(cellNum,OriNum+SfNum,2);
    G4_PeakSfLocListTotal = zeros(cellNum,OriNum+SfNum+2);
    %%The 1:12 is different Ori with Max Sf, 13:18 is different Sf with Max Ori


    for ci = 1:cellNum
        SfOri_Mean = squeeze(G4_SfOri_MeanSeListTotal(:,ci,:,1)); % [sf * ori]
        SfOri_Se = squeeze(G4_SfOri_MeanSeListTotal(:,ci,:,2)); % [sf * ori]
        SzLoc = squeeze(G4_SzLocListTotal(:,ci,:)); % [sf ori], sf * sz index
%         x=find(SfOri_Mean==max(max(SfOri_Mean)));
%         if isempty(x) 
%                    RawData_BadCell = [RawData_BadCell,ci];
%                    PeakSf = 1; PeakOri = 1;
%         else
            [PeakSf PeakOri]= find(SfOri_Mean==max(max(SfOri_Mean))); % max resp happens at which sf * ori?
%         end
        
        G4_RawDataforFITTINGListTotal(ci,1:OriNum,1) = SfOri_Mean(PeakSf,:); % ori resp at best sf band
        G4_RawDataforFITTINGListTotal(ci,OriNum+1:end,1) = (SfOri_Mean(:,PeakOri))'; % sf resp at best ori
        G4_RawDataforFITTINGListTotal(ci,1:OriNum,2) = SfOri_Se(PeakSf,:); % se
        G4_RawDataforFITTINGListTotal(ci,OriNum+1:end,2) = (SfOri_Se(:,PeakOri))';
        G4_PeakSfLocListTotal(ci,1:OriNum) = SzLoc(PeakSf,:); % peak index for sf*sz
        G4_PeakSfLocListTotal(ci,OriNum+1) = PeakOri; % 13 is peak orientation index
        G4_PeakSfLocListTotal(ci,OriNum+2:OriNum+SfNum+1) = SzLoc(:,PeakOri); % peak index for sf*sz of maximal ori
        G4_PeakSfLocListTotal(ci,OriNum+SfNum+2) = PeakSf;
    end


    save G4_RawDataforFITTINGListTotal.mat G4_RawDataforFITTINGListTotal;
    save G4_PeakSfLocListTotal.mat G4_PeakSfLocListTotal;


    %% Find Anova Data

    load G4_RspAvgOFFListTotal.mat; % [sf*sz neuron ori repeat]
    load G4_PeakSfLocListTotal.mat; % [neuron, peaksfindex + peak ori index + peaksfindex for each ori + peak sf index]

    cellNum = size(G4_RspAvgOFFListTotal,2);
    oln = size(G4_RspAvgOFFListTotal,3);
    sfn = SfNum;
    trialNum = size(G4_RspAvgOFFListTotal,4);
    Y1_AnovaListTotal = zeros(1,cellNum);

    for ci = 1:cellNum
        anovaList = zeros(oln+sfn,trialNum);
        for j3 = 1:oln
            Loc = G4_PeakSfLocListTotal(ci,j3); % peak index of sf*sz
            anovaList(j3,:)= G4_RspAvgOFFListTotal(Loc,ci,j3,:);
        end
        for j = oln+1:oln+sfn
            Loc2 = G4_PeakSfLocListTotal(ci,j+1); % max sz for each sf
            PeakOri = G4_PeakSfLocListTotal(ci,OriNum+1); % with max ori
            anovaList(j,:)= G4_RspAvgOFFListTotal(Loc2,ci,PeakOri,:);
        end    

        anovaList = anovaList';
        anova_test = kruskalwallis(anovaList(:,1:oln),1:oln,'off');
        p = anova_test(1);
        Y1_AnovaListTotal(1,ci) = p; %%%OriAnova

        anova_test2 = kruskalwallis(anovaList(:,oln+1:end),1:sfn,'off');
        p2 = anova_test2(1);
        F4_AnovaListTotal(1,ci) = p2; %%%SFAnova
    end

    save Y1_AnovaListTotal.mat Y1_AnovaListTotal;
    save F4_AnovaListTotal.mat F4_AnovaListTotal;

    %% direction pick

    load G4_PeakSfLocListTotal.mat;
    load G4_RspAvgOFFListTotal.mat;

    %Dir_Matrix = zeros(size(G4_PeakSfLocListTotal,1), 12);
    Dir_Matrix = zeros(size(G4_PeakSfLocListTotal,1), trialNum);

    for i = 1:size(G4_PeakSfLocListTotal,1)
        ci = i;
        PeakOri = G4_PeakSfLocListTotal(ci, OriNum+1); % peak ori index
        PeakSf = G4_PeakSfLocListTotal(ci, end); % peak sf index
        PeakNum = G4_PeakSfLocListTotal(ci, OriNum+1+PeakSf); % peak sf*sz index 
        Dir_Matrix(ci, :) = squeeze(G4_RspAvgOFFListTotal(PeakNum, ci, PeakOri, :)); % [neuron * 12], at best size, sf, ori
    end

    save Dir_Matrix Dir_Matrix;

    pvalueMat = zeros(size(Dir_Matrix,1),1);
    for i = 1:size(Dir_Matrix,1)
        p = ranksum(Dir_Matrix(i,1:trialNum/2), Dir_Matrix(i,trialNum/2+1:trialNum));
        %p = ranksum(Dir_Matrix(i,1:6), Dir_Matrix(i,7:12));
        pvalueMat(i) = p;
    end

    save pvalueMat pvalueMat;


    load G4_RspMeanTrialStdSeListTotal;
    withdir_index = find(pvalueMat<0.05); % very small


    SfSzNum = size(G4_RspAvgOFFListTotal,1);
    cellNum = size(G4_RspAvgOFFListTotal,2);
    oln = size(G4_RspAvgOFFListTotal,3);
    trialNum = trialNum/2;
    %1.Useful trial avg(<3sd); 2.Useful trial number; 3.Useful trial std; 4.Std/sqrt(trial number);
    for k = 1:length(withdir_index)
        ci = withdir_index(k);
        dirM = [mean(Dir_Matrix(ci,1:trialNum)), mean(Dir_Matrix(ci,trialNum+1:trialNum*2))];
        dirIdx = find(dirM == max(dirM));
        for ss = 1:SfSzNum
            for j3 = 1:oln %%Orientaion            
                rspTemporal = squeeze(G4_RspAvgOFFListTotal(ss,ci,j3,1+(dirIdx-1)*trialNum:trialNum+(dirIdx-1)*trialNum));
                rspMean = mean(rspTemporal);
                rspStd = std(rspTemporal);
                index = find(abs(rspTemporal-rspMean)>3*rspStd);    
                rspTemporal(index) = 0;     %%%%Dlete >3std trials
                rspTrial = length(find(rspTemporal~=0));
                rspMean = mean(rspTemporal(find(rspTemporal~=0)));
                rspStd = std(rspTemporal(find(rspTemporal~=0)));
                rspSe = rspStd./sqrt(rspTrial);
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,1) = rspMean;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,2) = rspTrial;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,3) = rspStd;
                G4_RspMeanTrialStdSeListTotal(ss,ci,j3,4) = rspSe;
            end
        end
    end

    save G4_RspMeanTrialStdSeListTotal.mat G4_RspMeanTrialStdSeListTotal; % recalculate this with preferred dir statistics
% stimuli = 1:1:144;
% errorbar(stimuli,G4_RspMeanTrialStdSeListTotal(1,:,1),G4_RspMeanTrialStdSeListTotal(1,:,4));
% ylim([-0.5 1])

    %% Find the Maximum Response in the matrix, redo this again
    %%%Pick the maximum response Size in each Sf*Ori condition(3to1)

    G4_SfOri_MeanSeListTotal = zeros(SfSzNum/SzNum,cellNum,oln,2);
    %1.Mean; 2.Se
    G4_SzLocListTotal = zeros(SfSzNum/SzNum,cellNum,oln);

    
    for ci = 1:cellNum
        SfSzNum_Oln_Mean = squeeze(G4_RspMeanTrialStdSeListTotal(:,ci,:,1));
        SfSzNum_Oln_Se = squeeze(G4_RspMeanTrialStdSeListTotal(:,ci,:,4));
            for j4 = 1:SfSzNum/3
                New_Mat = SfSzNum_Oln_Mean(j4*SzNum-SzNum+1:j4*SzNum,:);
                New_Mat_Se = SfSzNum_Oln_Se(j4*SzNum-SzNum+1:j4*SzNum,:);
%                 x = find(New_Mat==max(max(New_Mat)));
%                 x = x(1);
%                 [SzLoc,OriLoc] = ind2sub([SzNum,oln],find(x));
                x = find(New_Mat==max(max(New_Mat)));
                if isempty(x) 
                   RawData_BadCell = [RawData_BadCell,ci];
                   x=1;
                   SzLoc =1;OriLoc=1;
                else
                    [SzLoc,OriLoc] = find(New_Mat==max(max(New_Mat)));
                end
%                  [SzLoc,OriLoc]  = find(New_Mat==max(max(New_Mat)))
                G4_SfOri_MeanSeListTotal(j4,ci,:,1) = squeeze(New_Mat(SzLoc(1),:));
                G4_SfOri_MeanSeListTotal(j4,ci,:,2) = squeeze(New_Mat_Se(SzLoc(1),:));
                G4_SzLocListTotal(j4,ci,:) = j4*SzNum-SzNum+SzLoc(1);
            end
    end
    
    save G4_SfOri_MeanSeListTotal.mat G4_SfOri_MeanSeListTotal;
    save G4_SzLocListTotal.mat G4_SzLocListTotal;

    %% Pick the maximum Ori&Sf response

    load G4_SfOri_MeanSeListTotal.mat;
    load G4_SzLocListTotal.mat

    G4_RawDataforFITTINGListTotal = zeros(cellNum,sfn+oln,2);
    G4_PeakSfLocListTotal = zeros(cellNum,sfn+oln+2);
    %%The 1:12 is different Ori with Max Sf, 13:18 is different Sf with Max Ori


    for ci = 1:cellNum
        SfOri_Mean = squeeze(G4_SfOri_MeanSeListTotal(:,ci,:,1));
        SfOri_Se = squeeze(G4_SfOri_MeanSeListTotal(:,ci,:,2));
        SzLoc = squeeze(G4_SzLocListTotal(:,ci,:));
%         x=find(SfOri_Mean==max(max(SfOri_Mean)));
%         x=x(1);
        [PeakSf PeakOri]= find(SfOri_Mean==max(max(SfOri_Mean)));
        G4_RawDataforFITTINGListTotal(ci,1:oln,1) = SfOri_Mean(PeakSf,:); 
        G4_RawDataforFITTINGListTotal(ci,oln+1:end,1) = (SfOri_Mean(:,PeakOri))';
        G4_RawDataforFITTINGListTotal(ci,1:oln,2) = SfOri_Se(PeakSf,:);
        G4_RawDataforFITTINGListTotal(ci,oln+1:end,2) = (SfOri_Se(:,PeakOri))';
        G4_PeakSfLocListTotal(ci,1:oln) = SzLoc(PeakSf,:);
        if length(find(isnan(G4_RawDataforFITTINGListTotal(ci,:,1))==1))>0
            RawData_BadCell = [RawData_BadCell,ci];
        end
        G4_PeakSfLocListTotal(ci,oln+1) = PeakOri; %%13 is peak orientation
        G4_PeakSfLocListTotal(ci,oln+2:end-1) = SzLoc(:,PeakOri);
        G4_PeakSfLocListTotal(ci,end) = PeakSf;
    end


    save G4_RawDataforFITTINGListTotal.mat G4_RawDataforFITTINGListTotal;
    save G4_PeakSfLocListTotal.mat G4_PeakSfLocListTotal;


    %% Find Anova Data

    load G4_RspAvgOFFListTotal.mat;
    load G4_PeakSfLocListTotal.mat;
    load Y1_AnovaListTotal.mat;
    load F4_AnovaListTotal.mat;
    cellNum = size(G4_RspAvgOFFListTotal,2);
    oln = size(G4_RspAvgOFFListTotal,3);
    sfn = SfNum;
    %

    for k = 1:length(withdir_index)
        ci = withdir_index(k);
        dirM = [mean(Dir_Matrix(ci,1:trialNum)), mean(Dir_Matrix(ci,trialNum+1:trialNum*2))];
        dirIdx = find(dirM == max(dirM));
        anovaList = zeros(oln+sfn,trialNum);
        for j3 = 1:oln
            Loc = G4_PeakSfLocListTotal(ci,j3);
            anovaList(j3,:)= G4_RspAvgOFFListTotal(Loc,ci,j3,1+(dirIdx-1)*trialNum:trialNum+(dirIdx-1)*trialNum);
        end
        for j = oln+1:oln+sfn
            Loc2 = G4_PeakSfLocListTotal(ci,j+1);
            PeakOri = G4_PeakSfLocListTotal(ci,oln+1);
            anovaList(j,:)= G4_RspAvgOFFListTotal(Loc2,ci,PeakOri,1+(dirIdx-1)*trialNum:trialNum+(dirIdx-1)*trialNum);
        end    

        anovaList = anovaList';
        p = friedman(anovaList(:,1:oln), 1, 'off');
        Y1_AnovaListTotal(1,ci) = p; %%%OriAnova
        
        p = friedman(anovaList(:,oln+1:end), 1, 'off');
        F4_AnovaListTotal(1,ci) = p; %%%SFAnova
    end
    RawData_BadCell = [];
    RawData_BadCell = unique(RawData_BadCell);
    save Y1_AnovaListTotal.mat Y1_AnovaListTotal;
    save F4_AnovaListTotal.mat F4_AnovaListTotal;
    
    save RawData_BadCell.mat RawData_BadCell;

    %% save ori-tuned neuron label, according to .01 or .05
    targetcell_base5 = find(Y1_AnovaListTotal<.05);
    targetcell_base1 = find(Y1_AnovaListTotal<.01);
    targetcell_base10 = find(Y1_AnovaListTotal<.1);
    save targetcell_base5.mat targetcell_base5;
    save targetcell_base1.mat targetcell_base1;
    save targetcell_base10.mat targetcell_base10;


% % Anova
% temp = dir(pwd);
% load G4_RspAvgOFFListTotal.mat;
% 
%     cellNum = size(G4_RspAvgOFFListTotal, 1);
%     oln = size(G4_RspAvgOFFListTotal, 2);
%     trialNum = size(G4_RspAvgOFFListTotal, 3);
%     Y1_AnovaListTotal = zeros(1,cellNum);
% 
%     for ci = 1: cellNum
%         anovaList = zeros(oln, trialNum);
% %         anovaList = squeeze(G4_RspAvgOFFListTotal(ci,:,:));
%         anovaList = squeeze(G4_RspAvgOFFListTotal(ci,[1: 6]*5-1,:));
%         anovaList = anovaList'; %[repeat,ori]
%         p = friedman(anovaList, 1, 'off');
%         Y1_AnovaListTotal(ci) = p;
%     end
% 
%     save Y1_AnovaListTotal.mat Y1_AnovaListTotal;


 end









