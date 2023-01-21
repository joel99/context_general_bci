% For converting Pitt QL format to matlab objects (to then be ingested by my python libs..)
% Requires Pitt RTMA lib in matlab path

function prep_and_cast(root_dir, experiment, file)
% Load quicklogger trial, cast StimData to a struct, and save as processed matlab

src_file = fullfile(root_dir, experiment, file)
[data, iData] = prepData('files', [src_file])
% [iData] = prepData('files', [src_file], 'format_data', false) # doesn't work for some reason...
% data.StimData = struct(data.StimData)
% iData.StimData = struct(iData.StimData)
target_dir = fullfile(root_dir, "mat", experiment)
[dummy, file_stem, ext] = fileparts(src_file)
target_file = fullfile(target_dir, strcat(file_stem, ".mat"))
% save(target_file, 'iData')
save(target_file, 'iData', 'data')

end

