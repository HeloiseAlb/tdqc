#include "qudyn.h"
#include "system.h"
#include "algorithm"

SpinSystem::SpinSystem(bool displayOn):
    displayOn(displayOn) {

    id2 = cx_mat(zeros(2, 2), zeros(2, 2));
    id2(0, 0) = cx_double(1.0, 0.0);
    id2(1, 1) = cx_double(1.0, 0.0);

    // Here I use inverted order for spin up and down
    // This is because of how states are implemented
    //  in the qudyn library.

    sigmax = cx_mat(zeros(2, 2), zeros(2, 2));
    sigmax(0, 1) = cx_double(1.0, 0.0);
    sigmax(1, 0) = cx_double(1.0, 0.0);

    sigmay = cx_mat(zeros(2, 2), zeros(2, 2));
    // sigmay(0, 1) = cx_double(0.0, -1.0);
    // sigmay(1, 0) = cx_double(0.0, 1.0);
    sigmay(0, 1) = cx_double(0.0, 1.0);
    sigmay(1, 0) = cx_double(0.0, -1.0);

    sigmaz = cx_mat(zeros(2, 2), zeros(2, 2));
    // sigmaz(0, 0) = cx_double(1.0, 0.0);
    // sigmaz(1, 1) = cx_double(-1.0, 0.0);
    sigmaz(0, 0) = cx_double(-1.0, 0.0);
    sigmaz(1, 1) = cx_double(1.0, 0.0);
    pauliVector = {id2, sigmax, sigmay, sigmaz};
}


void SpinSystem::setCouplingMatrix() {
    couplingMatrix = zeros(systemSize, systemSize);
    // Long Range Ising
    // H = H_X + H_Z
    // H_X = 1/N sum_{l<m} s_l^x s_m^x/|m-l|^α + g sum_l s_l^x
    // H_Z = h sum_l s_l^z
    // g = hx, h = hz
    //
    // J sigma sigma = 4J S S
    // Upper triangular matrix
    for (unsigned i = 0; i < systemSize; ++i) {
        for (unsigned j = i + 1; j < systemSize; ++j) {
            // needs to be multiplied by j coefs
                couplingMatrix(i, j) = 4.0 / pow(j - i, alpha);
                // not needed but to be sure
                couplingMatrix(j, i) = 4.0 / pow(j - i, alpha);
        }
    }
}

void SpinSystem::setGates(
        const std::vector<double> &jList,
        const std::vector<std::vector<double>> &vhxList,
        const std::vector<std::vector<double>> &vhzList
        ) {
    // How to initialize the gate sequences

    // For a field:
    hxGates = std::vector<vec> (3*numberOfSteps);
    hzGates = std::vector<vec> (3*numberOfSteps);
    // For a coupling:
    jGates = std::vector<mat> (3*numberOfSteps);


    for (unsigned step = 0; step < numberOfSteps; ++step) {
        jGates[3 * step] = jList[step] * couplingMatrix;
        jGates[3 * step + 1] = zeros(systemSize, systemSize);
        jGates[3 * step + 2] = zeros(systemSize, systemSize);
        
        vec vhx = zeros(systemSize);
        vec vhz = zeros(systemSize);
        for (unsigned site = 0; site < systemSize; ++site){
            vhx(site) = 2.0 * vhxList[step][systemSize - 1 - site];
            vhz(site) = 2.0 * vhzList[step][systemSize - 1 - site];
        }
        if (gateOrder == "xz") {
            hxGates[3 * step] = zeros(systemSize);
            hxGates[3 * step + 1] = vhx;
            hxGates[3 * step + 2] = zeros(systemSize);

            hzGates[3 * step] = zeros(systemSize);
            hzGates[3 * step + 1] = zeros(systemSize);
            hzGates[3 * step + 2] = vhz;

        } else if (gateOrder == "zx") {
            hxGates[3 * step] = zeros(systemSize);
            hxGates[3 * step + 1] = zeros(systemSize);
            hxGates[3 * step + 2] = vhx;

            hzGates[3 * step] = zeros(systemSize);
            hzGates[3 * step + 1] = vhz;
            hzGates[3 * step + 2] = zeros(systemSize);
        }
    }
}


void SpinSystem::setInitialState(
    const std::vector<double> &stateReal, const std::vector<double> &stateImag
) {
    this->stateReal = stateReal;
    this->stateImag = stateImag;
}

double SpinSystem::start(const string &measurement) {

    QudynEngine qd(displayOn);
    // Start code
    qd.setProtocol("grid");
    qd.setTimeGridSteps(3 * numberOfSteps);
    qd.setSystemSize(systemSize);
    qd.setModel("spin");
    qd.setDiagonalization("lanczos");
    qd.setBoundaryCondition("open");
    qd.setTeType("complexAdaptive");
    qd.setMemorySaver(4);

    qd.setNumIterationsGrid(iterations);
    // qd.setNumIterationsGrid(50);
    qd.setInitialState(stateReal, stateImag);

    if (measurement == "energy") {
        addEnergyMeasurement(&qd);
    } else if (measurement == "particle_density") {
        addParticleDensityMeasurement(&qd);
    }
    
    if (entanglingGatesDir == "jx") {
        qd.setJxGrid(jGates);
    } else if (entanglingGatesDir == "jz") {
        qd.setJzGrid(jGates);
    }

    qd.setHzGrid(hzGates);
    qd.setHxGrid(hxGates);
    // Time evolution
    // Time duration of each of the gates, is set to 1
    vec timeGrid = zeros(3 * numberOfSteps + 1);
    // vec timeGrid = zeros(4 * numberOfSteps + 1);
    for (unsigned gate = 0; gate < 3 * numberOfSteps; ++gate) {
        timeGrid(gate+1) = timeGrid(gate) + 1.0;
    }
    qd.setTimeGrid(timeGrid);
    qd.setTimeSteps(timeSteps);
    qd.start();
    return getMeasurement(&qd, measurement);
}

double SpinSystem::getMeasurement(QudynEngine * qd, const string& measurement) {
    double measurementValue = 0.0;
    if (measurement == "fidelity") {
        QuDynDataBase * db = qd->getQuDynDataBase();
        std::vector<__m128d> finalState;
        finalState = * db->getState(db->getCurrentStateID());
        measurementValue = getFidelity(finalState);
    } else if (measurement == "energy" or measurement == "particle_density") {
        mat data = qd->getMeasurements()->
            getCustomMeasurementList()[0].getMeasurement();
        measurementValue = data(data.size() - 1);
    } else if (
            measurement == "trace_distance" || 
            measurement == "relative_entropy"
            ) {
        measurementValue = getDensityMatrixReward(qd, measurement);
    } else if (measurement == "loschmidt") {
        QuDynDataBase * db = qd->getQuDynDataBase();
        std::vector<__m128d> finalState;
        finalState = * db->getState(db->getCurrentStateID());
        unsigned hilbertSize = pow(2, systemSize);
        std::vector<double> finalStateReal (hilbertSize);
        std::vector<double> finalStateImag (hilbertSize);
        for (unsigned i; i < hilbertSize; ++i) {
            double vectorComp[2];
            // vectorComp[0]: real part
            // vectorComp[1]: imaginary part
            _mm_storeu_pd(&vectorComp[0], finalState[i]);
            finalStateReal[i] = vectorComp[0];
            finalStateImag[i] = vectorComp[1];
        }
        measurementValue = getFidelity(
                stateReal, stateImag,
                finalStateReal, finalStateImag
                );
    } else if (measurement == "staggered_magnetization") {
        measurementValue = getStaggeredMag(qd);
    } else if (measurement == "magnetization") {
        measurementValue = getMag(qd);
    } else if (measurement == "spin_fluctuations_zz") {
        measurementValue = getFluctuationZZ(qd);
    } else if (measurement == "entanglement_entropy") {
        Observables * obs = qd->getMeasurements()->getObservables();
        QuDynDataBase * db = qd->getQuDynDataBase();
        State state = State(db);
        db->copy(state.getStateID(), db->getCurrentStateID());
        mat obsmat = obs->entEntropy(&state);
        measurementValue = (obs->entEntropy(&state))(0, 0);
    } else {
        throw std::invalid_argument(
                "measurement `" + measurement + "` not recognized");
    }
    return measurementValue;
}

double SpinSystem::getStaggeredMag(
        QudynEngine * qd
        ) {
    QuDynDataBase * db = qd->getQuDynDataBase();
    Observables * obs = qd->getMeasurements()->getObservables();
    State state = State(db);

    db->copy(state.getStateID(), db->getCurrentStateID());
    valuesList svec = *(state.getValuesMap());

    double stagMag = 0.0;

    for (unsigned i = 0; i < systemSize; ++i) {
        vec hfield = zeros(systemSize);
        hfield[i] = 1.0;
        State statenew = State(db);
        valuesList svecnew;
        obs->sz(&statenew, &state, hfield); 
        svecnew = *(statenew.getValuesMap());
        stagMag += pow(-1, i) * real(scalarProduct(svecnew, svec));
        statenew.clear();
    }

    stagMag = stagMag / systemSize;
    state.clear();
    return stagMag;
}

double SpinSystem::getMag(
        QudynEngine * qd
        ) {

    QuDynDataBase * db = qd->getQuDynDataBase();
    Observables * obs = qd->getMeasurements()->getObservables();
    State state = State(db);

    db->copy(state.getStateID(), db->getCurrentStateID());

    valuesList svec = *(state.getValuesMap());

    double mag = 0.0;

    for (unsigned i = 0; i < systemSize; ++i) {
        vec hfield = zeros(systemSize);
        hfield[i] = 1.0;
        State statenew = State(db);
        valuesList svecnew;
        obs->sz(&statenew, &state, hfield); 
        svecnew = *(statenew.getValuesMap());
        mag += real(scalarProduct(svecnew, svec));
        statenew.clear();
    }

    mag = mag / systemSize;
    state.clear();
    return mag;
}


void SpinSystem::setTargetState(bool setRhoTarget) {
    QudynEngine qd(displayOn);
    qd.setProtocol("grid");
    unsigned nSteps = 10;
    qd.setTimeGridSteps(nSteps);
    qd.setSystemSize(systemSize);
    qd.setModel("spin");
    qd.setDiagonalization("lanczos");
    qd.setBoundaryCondition("open");
    qd.setTeType("complexAdaptive");
    qd.setMemorySaver(4);
    unsigned iterations_init = 50;
    qd.setNumIterationsGrid(iterations_init);
    qd.setInitialState(stateReal, stateImag);

    // Hamiltonian dependant step
    setGridForTargetState(&qd, nSteps);

    qd.setTimeSteps(timeSteps);
    qd.start();

    QuDynDataBase * db = qd.getQuDynDataBase();
    targetState = * db->getState(db->getCurrentStateID());
    
    if (setRhoTarget == true) {
        // indexDataBase * idb = db->getIndexDB();
        for (unsigned i = 0; i < systemSize; ++i) {
            for (unsigned j = i + 1; j < systemSize; ++j) {
                std::vector<unsigned> sites {i, j};
                rhoTarget[sites] = getTwoSiteReducedDensityMatrix(&qd, i, j);
            }
        }
    }
}


double SpinSystem::measurementTargetState(const string &measurement) {
    QudynEngine qd(displayOn);
    qd.setProtocol("grid");
    unsigned nSteps = 3;
    qd.setTimeGridSteps(nSteps);
    qd.setSystemSize(systemSize);
    qd.setModel("spin");
    qd.setDiagonalization("lanczos");
    qd.setBoundaryCondition("open");
    qd.setTeType("complexAdaptive");
    qd.setMemorySaver(4);
    qd.setNumIterationsGrid(iterations);
    
    // get the QudynEngine into the target state simply by starting from it
    // and acting with identity gates.
    unsigned hilbertSize = pow(2, systemSize);
    std::vector<double> targetStateReal (hilbertSize);
    std::vector<double> targetStateImag (hilbertSize);

    for (unsigned i; i < hilbertSize; ++i) {
        double vectorComp[2];
        // vectorComp[0]: real part
        // vectorComp[1]: imaginary part
        _mm_storeu_pd(&vectorComp[0], targetState[i]);
        targetStateReal[i] = vectorComp[0];
        targetStateImag[i] = vectorComp[1];
    }

    qd.setInitialState(targetStateReal, targetStateImag);

    if (measurement == "energy") {
        addEnergyMeasurement(&qd);
    }

    vec hGrid = zeros(nSteps);
    qd.setHxGrid(hGrid);
    // Time evolution
    // Time duration of each of the gates, is set to 1
    vec timeGrid = zeros(nSteps + 1);
    for (unsigned gate = 0; gate < nSteps; ++gate) {
        timeGrid(gate+1) = timeGrid(gate) + 1;
    }
    qd.setTimeGrid(timeGrid);
    qd.setTimeSteps(1);
    qd.start();

    return getMeasurement(&qd, measurement);
}


double SpinSystem::getFluctuationZZ(
        QudynEngine * qd
        ) {
    // calulcate quantum spin fluctuation in the middle of the chain
    //  = <Sz_m Sz_{m+1}> - <Sz_m><Sz_{m+1}>
    QuDynDataBase * db = qd->getQuDynDataBase();
    Observables * obs = qd->getMeasurements()->getObservables();
    State state = State(db);
    State state_2 = State(db);
    State state_3 = State(db);

    db->copy(state.getStateID(), db->getCurrentStateID());

    valuesList svec = *(state.getValuesMap());

    int midSite = systemSize / 2;
    vec hfield1 = zeros(systemSize);
    vec hfield2 = zeros(systemSize);
    hfield1[midSite] = 1.0;
    hfield2[midSite + 1] = 1.0;

    State statenew1 = State(db);
    State statenew2 = State(db);
    valuesList svecnew;
    obs->sz(&statenew1, &state, hfield1); 
    obs->sz(&statenew2, &statenew1, hfield2); 
    svecnew = *(statenew2.getValuesMap());
    double quantumCorr = real(scalarProduct(svecnew, svec));
    statenew1.clear();
    statenew2.clear();

    statenew1 = State(db);
    obs->sz(&statenew1, &state, hfield1); 
    svecnew = *(statenew1.getValuesMap());
    double classicalCorr = real(scalarProduct(svecnew, svec));
    statenew1.clear();

    statenew1 = State(db);
    obs->sz(&statenew1, &state, hfield2); 
    svecnew = *(statenew1.getValuesMap());
    classicalCorr *= real(scalarProduct(svecnew, svec));
    statenew1.clear();

    state.clear();
    return quantumCorr - classicalCorr;
}


double SpinSystem::getGroundStateEnergy() {
    QudynEngine qd(displayOn);
    qd.setProtocol("grid");
    unsigned nSteps = 10;
    // unsigned iterations = 100;
    qd.setTimeGridSteps(nSteps);
    qd.setSystemSize(systemSize);
    qd.setModel("spin");
    //
    // GS in sector with M = 0
    // only term with U(1) symmetry are considered (no error!)
    // qd.setModel("heisenberg");
    qd.setDiagonalization("lanczos");
    qd.setBoundaryCondition("open");
    qd.setTeType("complexAdaptive");
    qd.setMemorySaver(4);

    qd.setNumIterationsGrid(50);

    addEnergyMeasurement(&qd);
    qd.setInitialState("groundState");

    setInitialParameters(&qd);
    qd.setNumIterationsInitial(50);

    vec hGrid = zeros(nSteps);
    qd.setHxGrid(hGrid);
    // qd.setHzGrid(hGrid);

    // std::vector<mat> jGates = std::vector<mat> (nSteps);
    // for (unsigned gate = 0; gate < nSteps; ++gate) {
    //     mat J = zeros(systemSize, systemSize);
    //     jGates[gate] = J;
    // }
    // qd.setJxGrid(jGates);
    vec timeGrid = zeros(nSteps + 1);
    for (unsigned gate = 0; gate < nSteps; ++gate) {
        timeGrid(gate+1) = timeGrid(gate) + 1;
    }
    qd.setTimeGrid(timeGrid);
    qd.setTimeSteps(1);
    qd.start();
    
    mat data;
    data = qd.getMeasurements()->getCustomMeasurementList()[0].getMeasurement();
    // std::cout << data << std::endl;
    return data(data.size()-1);
}

double SpinSystem::getFidelity(const std::vector<__m128d> & state)  {
    return scalarProductNormSquare(state, targetState);
}

double SpinSystem::getFidelity(
        const std::vector<double> & real1,
        const std::vector<double> & imag1, 
        const std::vector<double> & real2, 
        const std::vector<double> & imag2
        ) {
    double result_real = 0.0;
    double result_imag = 0.0;
    for (unsigned i = 0; i < real1.size(); ++i) {
        result_real += (real1[i] * real2[i] + imag1[i] * imag2[i]);
        result_imag += (real1[i] * imag2[i] - imag1[i] * real2[i]);
    }
    return pow(result_real, 2) + pow(result_imag, 2);
}

cx_double SpinSystem::scalarProduct(const std::vector<__m128d> & state1,
        const std::vector<__m128d> & state2)
{
    // We calculate <state1, state2>
    double result_real = 0.0;
    double result_imag = 0.0;
    double vectorComp1[2];
    double vectorComp2[2];
    // vectorComp[0]: real part
    // vectorComp[1]: imaginary part
    for (unsigned i = 0; i < state1.size(); ++i) {
        _mm_storeu_pd(&vectorComp1[0], state1[i]);
        _mm_storeu_pd(&vectorComp2[0], state2[i]);
        result_real += (vectorComp1[0] * vectorComp2[0]
                        + vectorComp1[1] * vectorComp2[1]);
        result_imag += (vectorComp1[0] * vectorComp2[1]
                        - vectorComp1[1] * vectorComp2[0]);
    }

    return cx_double(result_real, result_imag);
}

double SpinSystem::scalarProductNormSquare(const std::vector<__m128d> & state1,
                               const std::vector<__m128d> & state2)
{
    // We calculate |<state1, state2>|^2

    cx_double product = scalarProduct(state1, state2);
    return pow(real(product), 2) + pow(imag(product), 2);

}

double SpinSystem::getDensityMatrixReward(
        QudynEngine * qd,
        const string & measurement
        )
{
    double sum = 0.0;
    double newVal = 0.0;
    // double weight = 0.0;
    double sumWeights = 0.0;
    // single-site RDM
    for (unsigned i = 0; i < systemSize; ++i) {
        for (unsigned j = i + 1; j < systemSize; ++j) {
            std::vector<unsigned> sites {i, j};
            cx_mat rho = getTwoSiteReducedDensityMatrix(qd, i, j);
            if (measurement == "trace_distance") 
                newVal = getTraceDistance(rho, rhoTarget[sites]);
            else if (measurement == "relative_entropy") {
                newVal = getRelativeEntropy(rho, rhoTarget[sites]);
            }
            if (averageExponent == -1.0) {
                sum = max(sum, newVal);
            } else {
                sum += pow(newVal, averageExponent);
                sumWeights += 1.0;
            }
            rho.clear();
        }
    }

    if (averageExponent != -1.0) {
        sum = sum / sumWeights;
    }
    return max(0.0, 1.0 - sum);
}

double SpinSystem::getRelativeEntropy(const cx_mat & m1, const cx_mat & m2)
{
    // return the relative entropy S(m1 || m2) where m1 and m2 are density
    // matrices: they must be hermitian positive semidefinite.
    double relativeEntropy;
    try {
        relativeEntropy = real(trace(m1 * (logmat(m1) - logmat(m2))));
    } catch (const std::runtime_error& e) {
        cout << "Calculating m1 * logmat(m1) using eigenvalues." << endl;
        relativeEntropy = -real(trace(m1 * logmat(m2)));
        vec eigv (eig_sym(m1));
        for (const double& ev : eigv){
            if (ev != 0.0 && !isnan(ev * log(ev))) {
                relativeEntropy += ev * log(ev);
            }
        }
    }
    // do something to make it work even when m1 and m2 have 0 eigenvalues
    // catch error: calculate eigenvalues explicitly  ... not same basis m1 and
    // m2...
    return relativeEntropy;
}

double SpinSystem::getTraceDistance(const cx_mat & m1, const cx_mat & m2)
{
    double normOfDiff = 0.0;
    // normOfDiff = std::abs(trace(m1 - m2));
    // normOfDiff = std::sqrt(std::abs(trace((m1 - m2).t() * (m1 - m2))));
    // if (norm_type == "trace_norm") {
    //     normOfDiff = std::norm(trace(m1 - m2));
    // }
    vec sval = svd(m1 - m2);
    normOfDiff = 0.5*sum(sval);
    return normOfDiff;
}

cx_mat SpinSystem::getTwoSiteReducedDensityMatrix(
        QudynEngine * qd,
        unsigned site1,
        unsigned site2
        )
{
    // maybe throw error if site1 == site2 ?
    // const int bitStateLength = 20;
    unsigned blockSize = 2;
    unsigned basisBlockSize = pow(2, blockSize);

    QuDynDataBase * db = qd->getQuDynDataBase();
    // indexDataBase * idb = db->getIndexDB();
    // qudynInt * iData = idb->data();
    // unsigned vectorSize = idb->size();


    // calculate <state| S^a_i S^b_j |state> to build the RDM
    // S^a = 1/2 Id2, S^x, S^y, S^z
    //
    Observables * obs = qd->getMeasurements()->getObservables();
    State state = State(db);
    db->copy(state.getStateID(), db->getCurrentStateID());
    valuesList svec = *(state.getValuesMap());

    vec hfield1 = zeros(systemSize);
    vec hfield2 = zeros(systemSize);
    hfield1[site1] = 1.0;
    hfield2[site2] = 1.0;

    std::vector<cx_mat> operators (4*4);
    std::vector<double> operatorsExpect (4*4);

    State state1 = State(db);
    State state2 = State(db);
    valuesList svecnew;

    double factor;
    for (unsigned i = 0; i < 4; ++i) {
        switch(i) {
            case 0: state1.copy(&state);
                    break;
            case 1: obs->sx(&state1, &state, hfield1);
                    break;
            case 2: obs->sy(&state1, &state, hfield1);
                    break;
            case 3: obs->sz(&state1, &state, hfield1);
                    break;
        }
        for (unsigned j = 0; j < 4; ++j) {
            switch(j) {
                case 0: state2.copy(&state1);
                        break;
                case 1: obs->sx(&state2, &state1, hfield2);
                        break;
                case 2: obs->sy(&state2, &state1, hfield2);
                        break;
                case 3: obs->sz(&state2, &state1, hfield2);
                        break;
            }
            svecnew = *(state2.getValuesMap());
            factor = 1.0;
            if (j == 0) factor *= 0.5;
            if (i == 0) factor *= 0.5;
            operatorsExpect[i+4*j] = factor *
                real(scalarProduct(svecnew, svec));
            operators[i+4*j] = kron(pauliVector[j], pauliVector[i]); 
        }
    }

    state.clear();
    state1.clear();
    state2.clear();

    cx_mat rho = cx_mat(zeros(basisBlockSize, basisBlockSize),
            zeros(basisBlockSize, basisBlockSize));

    // rho = sum_i <O_i> O_i
    // where any operator can be written as O = sum_i a_i O_i
    // and Tr(O_i O_k) = delta_ik
    // here : rho = sum_ij <sigma_i/2 sigma_j/2> sigma_i sigma_j

    for (unsigned i; i < 16; ++i) {
        rho = rho + operators[i] * operatorsExpect[i];
    }
    return rho;
}
