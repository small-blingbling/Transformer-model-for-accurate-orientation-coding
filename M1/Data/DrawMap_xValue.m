%% Draw Ori pinwheel
% x=cd;
% path = [cd '\Map\Ori pinwheel'];
colornum = 12;
colorOT = hsv(colornum);

increaseBrightness = @(color) min(color * 1.5, 1);

brighterColors = arrayfun(@(idx) increaseBrightness(colorOT(idx, :)), 1:colornum, 'UniformOutput', false);
brighterColors = vertcat(brighterColors{:});


colorOT = cat(1,brighterColors,[0.4 0.4 0.4])

load imY1;
sumIm(1:size(imY1,1),1:size(imY1,1),1) = imY1/2000;
sumIm(1:size(imY1,1),1:size(imY1,1),2) = imY1/2000;
sumIm(1:size(imY1,1),1:size(imY1,1),3) = imY1/2000;
% load sumIm.mat
% sumIm = sumIm/500;
load CCtotal.mat;
% load G4_PeakSfLocListTotal.mat
load xValueAtMax.mat
load targetcell_base10.mat
targetCell = targetcell_base10;
% load Y4_ParamList.mat;
Oricell = targetCell;
for i = 1:length(Oricell)
    P = CCtotal{Oricell(i)};
    % porie = Y4_ParamList(Oricell(i),2);
    % if porie<0
    % porie = porie+180;
    % elseif porie>180
    % porie = porie-180;
    % end
    % cidx = ceil(porie/15);
    cidx = round(xValueAtMax(Oricell(i)));
    for j = 1:length(P)
        id_x = mod(P(j),512);
        id_y = round((P(j) - id_x)/512)+1;
        %% if mod == 0; then should modify the results
        if(id_x==0);
            id_x = 512;
            id_y= id_y-1;
        end
        sumIm(id_x,id_y,:) = squeeze(colorOT(cidx, :));
    end
end

non_tuned_cell = setdiff([1:1720],targetCell);
Oricell = non_tuned_cell;
for i = 1:length(Oricell)
    P = CCtotal{Oricell(i)};
    % porie = Y4_ParamList(Oricell(i),2);
    % if porie<0
    % porie = porie+180;
    % elseif porie>180
    % porie = porie-180;
    % end
    % cidx = ceil(porie/15);
    cidx = xValueAtMax(Oricell(i));
    for j = 1:length(P)
        id_x = mod(P(j),512);
        id_y = round((P(j) - id_x)/512)+1;
        %% if mod == 0; then should modify the results
        if(id_x==0);
            id_x = 512;
            id_y= id_y-1;
        end
        sumIm(id_x,id_y,:) = [0.4 0.4 0.4];
    end
end


imshow(sumIm);
% title({[ 'CellNum = ' num2str(length(targetCell))]},'FontSize', 18, 'FontWeight', 'bold');
%saveas(gcf,[ReDir, '\Calculated\Ori_pinwheel0',num2str(idxn)], 'bmp');
colormap(colorOT);
colorbar('Ticks',[0.04:1/13:1], 'TickLabels',[0:15:180]);
%saveas(gcf,[path],'bmp');
% saveas(gcf,[ReDir, '\Calculated\Ori_pinwheel',num2str(idxn)], 'fig');
% pause
% close all;

% load sumIm.mat
% P = CCtotal{1,27};
% for j = 1:length(P)
%     id_x = mod(P(j),512);
%     id_y = round((P(j) - id_x)/512)+1;
%     %% if mod == 0; then should modify the results
%     if(id_x==0);
%         id_x = 512;
%         id_y= id_y-1;
%     end
%     sumIm(id_x,id_y,:) = squeeze(colorOT(1, :));
% end
% figure(2);
% imshow(sumIm);



