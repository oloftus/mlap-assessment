function data = extractDataFiles(dataFileName)

    rowFormat = '%10s,%d,%f';
    dateFormat = 'dd/mm/yyyy';
    fieldDelimiter = ':'; % Don't actually split the file on the correct delimiter
    eol = '\r\n';
    
    dataFile = fopen(dataFileName);
    dataRows = textscan(dataFile, rowFormat, 'Delimiter', fieldDelimiter, 'HeaderLines', 1, 'EndOfLine', eol);
    data = [double(datenum(dataRows{1}, dateFormat)), double(dataRows{2}), dataRows{3}];
    fclose(dataFile);
end
