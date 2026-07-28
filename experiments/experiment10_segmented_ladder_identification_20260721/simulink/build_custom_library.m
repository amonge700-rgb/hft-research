function build_custom_library()
%BUILD_CUSTOM_LIBRARY Compile the custom 8-winding Simscape component.

root = fileparts(mfilename('fullpath'));
old = pwd;
cleanup = onCleanup(@() cd(old));
cd(root);

ssc_build('hftlib');
fprintf('Built custom Simscape library: %s\n', fullfile(root, 'hftlib_lib.slx'));
end
