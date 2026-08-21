function componentFile = generate_full_L_component(nb, packageDir)
%GENERATE_FULL_L_COMPONENT Generate a fixed-size Simscape full-L component.
% The generated component implements v = R.*i + L*i.der for arbitrary
% symmetric positive-definite L supplied from the model workspace.

arguments
    nb (1,1) double {mustBeInteger,mustBePositive}
    packageDir (1,:) char
end

if ~exist(packageDir,'dir'), mkdir(packageDir); end
name = sprintf('coupled_inductor_fullL_%d',nb);
componentFile = fullfile(packageDir,[name '.ssc']);
fid = fopen(componentFile,'w');
assert(fid>=0,'Cannot create %s',componentFile);
c = onCleanup(@() fclose(fid));

fprintf(fid,'component %s\n',name);
fprintf(fid,'%% Full %dx%d reciprocal coupled winding component.\n',nb,nb);
fprintf(fid,'%% Constitutive law: v = R.*i + L*i.der\n\n');
fprintf(fid,'nodes\n');
for k=1:nb
    fprintf(fid,'    p%d = foundation.electrical.electrical;\n',k);
    fprintf(fid,'    n%d = foundation.electrical.electrical;\n',k);
end
fprintf(fid,'end\n\nparameters\n');
fprintf(fid,'    R = { ones(%d,1), ''Ohm'' };\n',nb);
fprintf(fid,'    L = { eye(%d), ''H'' };\n',nb);
fprintf(fid,'end\n\nvariables\n');
fprintf(fid,'    i = { zeros(%d,1), ''A'' };\n',nb);
fprintf(fid,'    v = { zeros(%d,1), ''V'' };\n',nb);
fprintf(fid,'end\n\nbranches\n');
for k=1:nb
    fprintf(fid,'    i(%d) : p%d.i -> n%d.i;\n',k,k,k);
end
fprintf(fid,'end\n\nequations\n    v == [ ');
for k=1:nb
    if k>1, fprintf(fid,'; '); end
    fprintf(fid,'p%d.v-n%d.v',k,k);
end
fprintf(fid,' ];\n    v == R.*i + L*i.der;\nend\n\nend\n');
end

