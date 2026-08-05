#ifndef SYSTEM_H 
#define SYSTEM_H

#include "qudyn.h"

class SpinSystem {
public:
    SpinSystem(bool displayOn = false);

    void setCouplingMatrix();
    void setGates(
            const std::vector<double> &jxList,
            const std::vector<std::vector<double>> &vhxList,
            const std::vector<std::vector<double>> &vhzList
            );

    void setInitialState(
        const std::vector<double> &stateReal,
        const std::vector<double> &stateImag
    );

    // functions that need the explicit Hamiltonian
    virtual void addEnergyMeasurement(QudynEngine * qd) = 0;
    virtual void setInitialParameters(QudynEngine * qd) = 0;
    virtual void setGridForTargetState(QudynEngine * qd, unsigned nSteps) = 0;
    // only for Schwinger model, but defined as pure virtual for pybind
    // (non-pure virtual methods cause trouble) 
    virtual void addParticleDensityMeasurement(QudynEngine * qde) = 0;

    void setTargetState(bool setRhoTarget);
    double measurementTargetState(const string &measurement);

    double start(const string &measurement);
    double getMeasurement(QudynEngine * qd, const std::string& measurement);

    double getGroundStateEnergy();
    double getFidelity(const std::vector<__m128d> & state);
    double getFidelity(
            const std::vector<double> & real1,
            const std::vector<double> & imag1,
            const std::vector<double> & real2,
            const std::vector<double> & imag2
            );

    cx_double scalarProduct(const std::vector<__m128d> & state1,
            const std::vector<__m128d> & state2);

    double scalarProductNormSquare(const std::vector<__m128d> & state1,
                                   const std::vector<__m128d> & state2);

    double getDensityMatrixReward(
            QudynEngine * qd,
            const string & measurement
            );

    double getTraceDistance(const cx_mat & m1, const cx_mat & m2);
    double getRelativeEntropy(const cx_mat & m1, const cx_mat & m2);

    cx_mat getTwoSiteReducedDensityMatrix(
            QudynEngine * qd,
            unsigned site1,
            unsigned site2
            );

    // double getStaggeredMag(const std::vector<__m128d> & state);
    double getStaggeredMag(QudynEngine * qd);
    double getMag(QudynEngine * qd);
    double getFluctuationZZ(QudynEngine * qd);


// private:
protected:
    bool displayOn;
    unsigned systemSize;
    double timeSegment;
    unsigned numberOfSteps;

    unsigned timeSteps;
    unsigned iterations;


    string gateOrder;
    string entanglingGatesDir;
    double alpha;

    string measurement;
    double averageExponent;
    bool periodicBoundaryConditions;

    // initial state
    std::vector<double> stateReal;
    std::vector<double> stateImag;


    std::vector<__m128d> targetState;
    std::map< std::vector<unsigned>, cx_mat> rhoTarget;

    mat couplingMatrix;

    cx_mat id2;
    cx_mat sigmax;
    cx_mat sigmay;
    cx_mat sigmaz; 
    std::vector<cx_mat> pauliVector;

    // vectors of gates used for the digitial simulations
    // For a field:
    std::vector<vec> hxGates;
    std::vector<vec> hzGates;
    std::vector<vec> hyGates;
    // For a coupling:
    std::vector<mat> jGates;
};

#endif // SYSTEM_H
