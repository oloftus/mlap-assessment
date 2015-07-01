function inputVector = generateFeatures(companyData, sectorData, knotPoints, featureSelection)

    numDataRows = size(companyData, 1);

    onez = ones(numDataRows, 1);

    companyPrices = companyData(:,3);

    companyDates = companyData(:,1) - companyData(1,1); % Change date offset to 0
    companyDatesPower2 = companyDates .^ 2;
    companyDatesPower3 = companyDates .^ 3;
    
    companyDayNums = weekday(companyData(:,1));
    
    companyVolumes = companyData(:,2);
    companyVolumesDay1 = companyVolumes(1,1);
    companyVolumesMinus1Day = [companyVolumesDay1; companyVolumes(1:end - 1,:)];
    
    sectorVolumes = sectorData(:,2);
    sectorVolumesDay1 = sectorVolumes(1,1);
    sectorVolumesMinus1Day = [sectorVolumesDay1; sectorVolumes(1:end - 1,:)];

    sectorPrices = sectorData(:, 3);
    sectorPricesDay1 = sectorPrices(1,1);
    sectorPricesMinus1Day = [sectorPricesDay1; sectorPrices(1:end - 1,:)];

    companyPriceDay1 = companyPrices(1,1);
    companyPricesMinus1Day = [companyPriceDay1; companyPrices(1:end - 1,:)];
    companyPricesMinus2Day = [companyPriceDay1; companyPriceDay1; companyPrices(1:end - 2,:)];
    companyPricesMinus3Day = [companyPriceDay1; companyPriceDay1; companyPriceDay1; companyPrices(1:end - 3,:)];
    companyPricesMinus4Day = [companyPriceDay1; companyPriceDay1; companyPriceDay1; companyPriceDay1; companyPrices(1:end - 4,:)];
    companyPricesMinus5Day = [companyPriceDay1; companyPriceDay1; companyPriceDay1; companyPriceDay1; companyPriceDay1;companyPrices(1:end - 5,:)];
    companyPriceGradient1 = companyPricesMinus2Day - companyPricesMinus1Day;
    companyPriceGradient2 = companyPricesMinus3Day - companyPricesMinus2Day;
    companyPriceGradient3 = companyPricesMinus4Day - companyPricesMinus3Day;
    companyPriceGradient4 = companyPricesMinus5Day - companyPricesMinus4Day;
    companyPriceAvgGradient = mean([companyPriceGradient1, companyPriceGradient2, companyPriceGradient3, companyPriceGradient4], 2);
    
    sectorPriceD1 = sectorPricesMinus1Day(1,1);
    sectorPricesMinus1Day = [sectorPriceD1; sectorPricesMinus1Day(1:end - 1,:)];
    sectorPricesMinus2Day = [sectorPriceD1; sectorPriceD1; sectorPricesMinus1Day(1:end - 2,:)];
    sectorPricesMinus3Day = [sectorPriceD1; sectorPriceD1; sectorPriceD1; sectorPricesMinus1Day(1:end - 3,:)];
    sectorPricesMinus4Day = [sectorPriceD1; sectorPriceD1; sectorPriceD1; sectorPriceD1; sectorPricesMinus1Day(1:end - 4,:)];
    sectorPricesMinus5Day = [sectorPriceD1; sectorPriceD1; sectorPriceD1; sectorPriceD1; sectorPriceD1;sectorPricesMinus1Day(1:end - 5,:)];
    sectorPriceGradient1 = sectorPricesMinus2Day - sectorPricesMinus1Day;
    sectorPriceGradient2 = sectorPricesMinus3Day - sectorPricesMinus2Day;
    sectorPriceGradient3 = sectorPricesMinus4Day - sectorPricesMinus3Day;
    sectorPriceGradient4 = sectorPricesMinus5Day - sectorPricesMinus4Day;
    sectorPriceAvgGradient = mean([sectorPriceGradient1, sectorPriceGradient2, sectorPriceGradient3, sectorPriceGradient4], 2);
    
    knotPointsFeature = max(0, repmat(companyDates, 1, size(knotPoints, 2)) - repmat(knotPoints, size(companyDates, 1), 1));
    
    inputVector = [onez, companyDates, companyPricesMinus1Day, companyDayNums, companyVolumesMinus1Day, sectorVolumesMinus1Day...
            companyPriceAvgGradient, sectorPricesMinus1Day, sectorPriceAvgGradient, companyDatesPower2, companyDatesPower3, knotPointsFeature];
    
    for i = 2:size(inputVector, 2)
        vec = inputVector(:,i);
        inputVector(:,i) = vec ./ max(vec);
    end
    inputVector(isnan(inputVector)) = 0;
    
    inputVector = inputVector .* repmat(featureSelection', size(inputVector,1), 1);
    
end

