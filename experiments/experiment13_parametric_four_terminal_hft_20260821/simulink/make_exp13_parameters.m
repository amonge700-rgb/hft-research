function p = make_exp13_parameters(Np,Ns,kind)
%MAKE_EXP13_PARAMETERS Create reproducible synthetic parameters.
% kind='regression4' reproduces the Experiment-10 winding parameters.
% kind='candidate8' creates a general SPD L and banded nonlocal Cps.

if nargin<3, kind='candidate8'; end
assert(Np>0 && Ns>0 && mod(Np,1)==0 && mod(Ns,1)==0);
nb=Np+Ns;
p=struct('Np',Np,'Ns',Ns,'name',kind);

switch lower(kind)
    case 'regression4'
        assert(Np==4 && Ns==4,'regression4 requires Np=Ns=4');
        ratio=4; Lm=220e-6;
        p.R=[repmat(0.18/Np,Np,1);repmat((0.18/ratio^2)/Ns,Ns,1)];
        leak=[repmat(3.6e-6/Np,Np,1);repmat((3.6e-6/ratio^2)/Ns,Ns,1)];
        a=[ones(Np,1);-ones(Ns,1)/ratio];
        p.L=diag(leak)+(Lm/Np^2)*(a*a.');
        p.Cseries=[repmat(72e-12,Np,1);repmat(42e-12,Ns,1)];
        p.Cg_p=[7;6;5;4;0]*1e-12;
        p.Cg_s=[5;4.5;4;3.5;0]*1e-12;
        p.Cps=diag(repmat(28.36e-12/Np,Np,1));
    case 'candidate8'
        assert(Np==8 && Ns==8,'candidate8 currently defines the 8+8 candidate');
        ratio=4; Lm=220e-6;
        rp=0.18/Np*(1+0.12*cos((1:Np)'*pi/(Np+1)));
        rs=(0.18/ratio^2)/Ns*(1+0.10*sin((1:Ns)'*pi/(Ns+1)));
        p.R=[rp;rs];
        leak=[3.6e-6/Np*(1+0.18*cos((1:Np)'*pi/(Np+1))); ...
              (3.6e-6/ratio^2)/Ns*(1+0.15*sin((1:Ns)'*pi/(Ns+1)))];
        modes=zeros(nb,3);
        modes(:,1)=[ones(Np,1);-ones(Ns,1)/ratio];
        modes(:,2)=[sin((1:Np)'*pi/(Np+1));-0.22*sin((1:Ns)'*pi/(Ns+1))];
        modes(:,3)=[cos((1:Np)'*pi/(Np+1));0.18*cos((1:Ns)'*pi/(Ns+1))];
        p.L=diag(leak)+(Lm/Np^2)*(modes(:,1)*modes(:,1).') ...
            +0.55e-6*(modes(:,2)*modes(:,2).') ...
            +0.25e-6*(modes(:,3)*modes(:,3).');
        p.Cseries=[72e-12*(1+0.10*sin((1:Np)'*pi/(Np+1))); ...
                   42e-12*(1+0.08*cos((1:Ns)'*pi/(Ns+1)))];
        p.Cg_p=linspace(7,2.5,Np+1)'*1e-12;
        p.Cg_s=linspace(5,2.0,Ns+1)'*1e-12;
        p.Cps=zeros(Np,Ns);
        for i=1:Np
            for j=1:Ns
                d=abs(i-j);
                if d<=2
                    p.Cps(i,j)=(28.36e-12/Np)*exp(-0.9*d);
                end
            end
        end
    otherwise
        error('Unknown parameter kind: %s',kind);
end

p.L=(p.L+p.L.')/2;
assert(all(p.R>0) && all(p.Cseries>=0));
assert(isequal(size(p.L),[nb nb]));
assert(min(eig(p.L))>0,'L must be positive definite');
assert(isequal(size(p.Cps),[Np Ns]));
end

