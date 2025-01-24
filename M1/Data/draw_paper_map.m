%% Draw Ori pinwheel
% x=cd;
% path = [cd '\Map\Ori pinwheel'];
colornum = 12;
colorOT = hsv(colornum);

load imY1;
sumIm(1:size(imY1,1),1:size(imY1,1),1) = imY1/2000;
sumIm(1:size(imY1,1),1:size(imY1,1),2) = imY1/2000;
sumIm(1:size(imY1,1),1:size(imY1,1),3) = imY1/2000;
% load sumIm.mat
% sumIm = sumIm/500;
load CCtotal.mat;
load G4_PeakSfLocListTotal.mat
for o = 1:12
    figure(o);
    load imY1;
    sumIm(1:size(imY1,1),1:size(imY1,1),1) = imY1/2000;
    sumIm(1:size(imY1,1),1:size(imY1,1),2) = imY1/2000;
    sumIm(1:size(imY1,1),1:size(imY1,1),3) = imY1/2000;
    targetCell = find(G4_PeakSfLocListTotal(:,13)==o);
    Oricell = targetCell;
    for i = 1:length(Oricell)
       
        P = CCtotal{Oricell(i)};
      
        cidx = G4_PeakSfLocListTotal(Oricell(i), 13);
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
    imshow(sumIm);
end

% title({[ 'CellNum = ' num2str(length(Oricell))]},'FontSize', 18, 'FontWeight', 'bold');
%saveas(gcf,[ReDir, '\Calculated\Ori_pinwheel0',num2str(idxn)], 'bmp');
% colormap(colorOT);
% colorbar('Ticks',[0:1/12:1], 'TickLabels',[0:15:180]);
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

load order_fig_1st_100.mat


for i = 61:72
    figure(i);
    imagesc(squeeze(order_fig_1st_100(i,:,:)));
    colormap('gray');
    axis off;
end

