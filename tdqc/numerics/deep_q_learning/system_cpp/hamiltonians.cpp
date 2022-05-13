#include "hamiltonians.h"

LongRangeIsing::LongRangeIsing(bool displayOn): SpinSystem(displayOn) {}

void LongRangeIsing::setSystem(
        const unsigned &systemSize,
        const unsigned &numberOfSteps,
        const double &jx,
        const double &hx, 
        const double &hz,
        const double &alpha,
        const double &timeSegment,
        const string &gateOrder,
        const string &entanglingGatesDir,
        const double & averageExponent,
        const bool & periodicBoundaryConditions
        ) {

    // Parameters
    this->systemSize = systemSize;
    this->jx = jx;
    this->hx = hx;
    this->hz = hz;
    this->gateOrder = gateOrder;
    this->entanglingGatesDir = entanglingGatesDir;

    this->alpha = alpha;
    this->timeSegment = timeSegment;
    this->numberOfSteps = numberOfSteps;

    this->averageExponent = averageExponent;

    setCouplingMatrix();

    this->hamiltonianCouplingMatrix = jx * couplingMatrix;


    this->timeSteps = 1;
    // Parameter controlling the accuracy of Lanczos
    this->iterations = 26;

}

void LongRangeIsing::addEnergyMeasurement(QudynEngine * qde) {
    CustomMeasurement msrmnt;
    msrmnt.setSystemSize(systemSize);
    msrmnt.setNameId("Energy"); //Some identifier
    //  defined with 1/2 factor? 1/N factor?
    //  -> define on spin-1/2 operator: h * sigma = 2h * S
    vec vhz = 2.0 * hz * ones(systemSize);
    vec vhx = 2.0 * hx * ones(systemSize);
    msrmnt.setFieldZ(vhz);
    msrmnt.setFieldX(vhx);
    msrmnt.setCouplingsX(hamiltonianCouplingMatrix);
    qde->addCustomMeasurement(msrmnt);
}


void LongRangeIsing::setInitialParameters(QudynEngine * qde) {
    vec vhz = 2.0 * hz * ones(systemSize);
    vec vhx = 2.0 * hx * ones(systemSize);
    qde->setHxInitial(vhx);
    qde->setHzInitial(vhz);
    qde->setJxInitial(hamiltonianCouplingMatrix);
}


void LongRangeIsing::setGridForTargetState(QudynEngine * qde, unsigned nSteps) {
    vec timeGrid = zeros(nSteps + 1);
    for (unsigned gate = 0; gate < nSteps; ++gate) {
        timeGrid(gate+1) = timeGrid(gate) + timeSegment;
    }
    qde->setTimeGrid(timeGrid);

    vec vhz = 2.0 * hz * ones(systemSize);
    vec vhx = 2.0 * hx * ones(systemSize);
    // For a field:
    std::vector<vec> hxGates = std::vector<vec> (nSteps);
    std::vector<vec> hzGates = std::vector<vec> (nSteps);
    // For a coupling:
    std::vector<mat> jxGates = std::vector<mat> (nSteps);
    jxGates[0] = hamiltonianCouplingMatrix;
    hxGates[0] = vhx;
    hzGates[0] = vhz;
    vec vhzeros = zeros(systemSize);
    mat Jzeros = zeros(systemSize, systemSize);
    for (unsigned gate = 1; gate < nSteps; ++gate) {
        jxGates[gate] = Jzeros;
        hxGates[gate] = vhzeros;
        hzGates[gate] = vhzeros;
    }
    qde->setJxGrid(jxGates);
    qde->setHxGrid(hxGates);
    qde->setHzGrid(hzGates);
}

// ill-defined, but needed for compatibility with pybind
// non-pure virtual methods cause trouble: runtime error when importing
// also, needs to be defined (not just declared)
void LongRangeIsing::addParticleDensityMeasurement(QudynEngine * qde) {}


Schwinger::Schwinger(bool displayOn): SpinSystem(displayOn) {}

void Schwinger::setSystem(
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
        ) {

    // Parameters
    this->systemSize = systemSize;
    this->m_coupling = m_coupling;
    this->w_coupling = w_coupling;
    this->j_coupling = j_coupling;

    this->gateOrder = gateOrder;
    this->entanglingGatesDir = entanglingGatesDir;

    this->averageExponent = averageExponent;

    this->alpha = alpha;
    this->timeSegment = timeSegment;
    this->numberOfSteps = numberOfSteps;


    setCouplingMatrix();
    hamiltonianCouplingMatrixXY = zeros(systemSize, systemSize);
    // hamiltonianCouplingMatrixZ = zeros(systemSize, systemSize);
    // hamiltonianFieldZ = zeros(systemSize);
    //
    // Schwinger
    // H = Hpm + Hz
    //
    // H_pm = w * sum_l [sigma_l^+ sigma_{l+1}^- + sigma_l^- sigma_{l+1}^+]
    // Hz = m/2 sum_l=1^N (-1)^l + j sum_l L_l^2
    // with L_l = 1/2 sum_i=1^l (sigma_i^z + (-1)^i)
    //
    // --> Hz = sum_l=1^{N}[ m/2 (-1)^{l} sigma^z_l
    //                       + j/2 sum_i=1^l sigma_i^z * (-1 if l odd, 0 else)
    //                       + j/2 sum_{j>i=1}^l sigma_i^z sigma_j^z ]
    //
    // also, sigma^a = 2 S^a
    //
    // Upper triangular matrix
    for (unsigned i = 0; i < systemSize - 1; ++i) {
        hamiltonianCouplingMatrixXY(i, i+1) = 2.0 * w_coupling;
    }

    setHamiltonianCouplingMatrixZ(hamiltonianCouplingMatrixZ);
    setHamiltonianFieldZ(hamiltonianFieldZ);

    this->timeSteps = 1;
    // Parameter controlling the accuracy of Lanczos
    this->iterations = 26;
}

void Schwinger::setHamiltonianCouplingMatrixZ(mat &couplingMatrix) {
    //reminder: sites in reversed order to be consistent with the Python code
    couplingMatrix = zeros(systemSize, systemSize);
    for (unsigned l = 0; l < systemSize; ++l) {
        for (unsigned i = 0; i <= l; ++i) {
            for (unsigned j = i + 1; j <= l; ++j) {
                couplingMatrix(systemSize - 1 - j, systemSize - 1 - i) 
                    += 2 * j_coupling;
            }
        }
    }
}

void Schwinger::setHamiltonianFieldZ(vec &field) {
    field = zeros(systemSize);
    for (unsigned l = 0; l < systemSize; ++l) {
        field(systemSize - 1 - l) += m_coupling * pow(-1.0, l + 1);
        for (unsigned i = 0; i <= l; ++i) {
            // if (l+1) is odd (l is even): 
            // sum_{j=0, l} (-1)^{j+1} = sum_{j=1, l+1} (-1)^j // = -1, else = 0
            if (l % 2 == 0) {
                field(systemSize - 1 - i) -= j_coupling;
            }
        }
    }
}


void Schwinger::addEnergyMeasurement(QudynEngine * qde) {
    CustomMeasurement msrmnt;
    msrmnt.setSystemSize(systemSize);
    msrmnt.setNameId("Energy"); //Some identifier
    msrmnt.setCouplingsXY(hamiltonianCouplingMatrixXY);
    msrmnt.setCouplingsZ(hamiltonianCouplingMatrixZ);
    msrmnt.setFieldZ(hamiltonianFieldZ);
    qde->addCustomMeasurement(msrmnt);
}

void Schwinger::addParticleDensityMeasurement(QudynEngine * qde) {
    CustomMeasurement msrmnt;
    msrmnt.setSystemSize(systemSize);
    msrmnt.setNameId("Particle Number Density"); //Some identifier

    // even and odd sites are not equivalent.
    // This should be the same field as defined in the Hamiltonian
    vec field = zeros(systemSize);
    for (unsigned l = 0; l < systemSize; ++l) {
        field(systemSize - 1 - l) += m_coupling * pow(-1.0, l + 1);
        }
    msrmnt.setCouplingsZ(field);
    qde->addCustomMeasurement(msrmnt);
}


void Schwinger::setInitialParameters(QudynEngine * qde) {
    qde->setJxyInitial(hamiltonianCouplingMatrixXY);
    qde->setJzInitial(hamiltonianCouplingMatrixZ);
    qde->setHzInitial(hamiltonianFieldZ);
}


void Schwinger::setGridForTargetState(QudynEngine * qde, unsigned nSteps) {
    vec timeGrid = zeros(nSteps + 1);
    for (unsigned gate = 0; gate < nSteps; ++gate) {
        timeGrid(gate+1) = timeGrid(gate) + timeSegment;
    }
    qde->setTimeGrid(timeGrid);

    // For a field:
    hzGates = std::vector<vec> (nSteps);
    // For a coupling:
    std::vector<mat> jxyGates = std::vector<mat> (nSteps);
    std::vector<mat> jzGates = std::vector<mat> (nSteps);
    hzGates[0] = hamiltonianFieldZ;
    jxyGates[0] = hamiltonianCouplingMatrixXY;
    jzGates[0] = hamiltonianCouplingMatrixZ;
    vec hzeros = zeros(systemSize);
    mat jzeros = zeros(systemSize, systemSize);
    std::vector<mat> jGates = std::vector<mat> (nSteps);
    for (unsigned gate = 1; gate < nSteps; ++gate) {
        jxyGates[gate] = jzeros;
        jzGates[gate] = jzeros;
        hzGates[gate] = hzeros;
    }
    qde->setJxyGrid(jxyGates);
    qde->setJzGrid(jzGates);
    qde->setHzGrid(hzGates);
}
