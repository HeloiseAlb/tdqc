#ifndef HAMILTONIANS_H 
#define HAMILTONIANS_H

#include "system.h"

class LongRangeIsing : public SpinSystem {
public:
    LongRangeIsing(bool displayOn = false);

    void setSystem(
            const unsigned &systemSize,
            const unsigned &numberOfSteps,
            const double &jx,
            const double &hx, 
            const double &hz,
            const double &alpha,
            const double &timeSegment,
            const string &gateOrder,
            const string &entanglingGatesDir,
            const double &averageExponent,
            const bool & periodicBoundaryConditions
            );


    void addEnergyMeasurement(QudynEngine * qde) override;
    void setInitialParameters(QudynEngine * qde) override;
    void setGridForTargetState(QudynEngine * qde, unsigned nSteps) override;

    // not usable, but needed for compatibility with pybind
    // (non-pure virtual methods cause trouble) 
    void addParticleDensityMeasurement(QudynEngine * qde) override;
private:
    double jx;
    double hx;
    double hz;

    mat hamiltonianCouplingMatrix;
};


class Schwinger : public SpinSystem {
public:
    Schwinger(bool displayOn = false);

    void setSystem(
            const unsigned &systemSize,
            const unsigned &numberOfSteps,
            const double &m_coupling,
            const double &w_coupling,
            const double &j_coupling,
            const double &alpha,
            const double &timeSegment,
            const string &gateOrder,
            const string &entanglingGatesDir,
            const double & averageExponent
            );

    void addEnergyMeasurement(QudynEngine * qde) override;
    void setInitialParameters(QudynEngine * qde) override;
    void setGridForTargetState(QudynEngine * qde, unsigned nSteps) override;

    void addParticleDensityMeasurement(QudynEngine * qde) override;

    void setHamiltonianCouplingMatrixZ(mat & couplingMatrix);
    void setHamiltonianFieldZ(vec & field);


private:
    double m_coupling;
    double w_coupling;
    double j_coupling;
    mat hamiltonianCouplingMatrixXY;
    mat hamiltonianCouplingMatrixZ;
    vec hamiltonianFieldZ;
};

#endif // HAMILTONIANS_H
