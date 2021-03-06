
if exist('../eigen/Eigen/Dense','file')
    disp('compiling ...');
    mex -I./../eigen ...
        ../src/Meta.cpp ../src/ISolver.cpp ...
        ../src/GradientDescentSolver.cpp ...
        ../src/ConjugateGradientSolver.cpp ...
        ../src/NewtonDescentSolver.cpp ...
        ../src/BfgsSolver.cpp ...
        ../src/LbfgsSolver.cpp   ...
        ../src/LbfgsbSolver.cpp ...
        cppsolver.cpp  ...
        CXXFLAGS="\$CXXFLAGS -std=c++11" ...
        COPTIMFLAGS="\$COPTIMFLAGS -O2" ...
        -output cppsolver;
else
    disp('Eigen is missing');
end


