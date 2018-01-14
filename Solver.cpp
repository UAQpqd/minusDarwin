//
// Created by dev on 1/13/2018.
//

#include "Solver.hpp"


MinusDarwin::Solver::Solver(const MinusDarwin::SolverParameterSet &t_sParams,
                            const std::function<float(Agent)> &t_scoreFunction) :
        Solver(t_sParams, t_scoreFunction, bc::device()) {

}

MinusDarwin::Solver::Solver(const MinusDarwin::SolverParameterSet &t_sParams,
                            const std::function<float(Agent)> &t_scoreFunction, const bc::device &t_device
) :
        scoreFunction(t_scoreFunction), sParams(t_sParams),
        device(t_device) {

}

void MinusDarwin::Solver::evaluatePopulation(std::vector<float> &scores, MinusDarwin::Population &p) {
    std::transform(p.begin(),p.end(),scores.begin(),[this] (const Agent &a) { return evaluateAgent(a); });
}

float MinusDarwin::Solver::evaluateAgent(const MinusDarwin::Agent &a) {
    return scoreFunction(a);
}

void MinusDarwin::Solver::initPopulation(MinusDarwin::Population &p) {
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_real_distribution<float> urd(0.0f, 1.0f);
    for(auto &a : p) for(auto &v : a) v = urd(re);
}

MinusDarwin::Agent MinusDarwin::Solver::run(bool verbose) {
    Population X(sParams.popSize,std::vector<float>(sParams.dims,0.0f));
    Population Y(sParams.popSize,std::vector<float>(sParams.dims,0.0f));
    std::vector<float> Xscores(sParams.popSize);
    std::vector<float> Yscores(sParams.popSize);
    initPopulation(X);
    evaluatePopulation(Xscores,X);
    bool goalReached = checkEpsilonReached(Xscores);
    for(size_t g = 0; g<sParams.maxGens && !goalReached; g++) {
        crossoverPopulation(X,Y,sParams.mode==CrossoverMode::Best?getBestAgentId(Xscores):0);
        selectionPopulation(X,Y);
        goalReached = checkEpsilonReached(Xscores);
    }
    return X.at(getBestAgentId(Xscores));
}

size_t MinusDarwin::Solver::getBestAgentId(const std::vector<float> &scores) {
    return std::distance(scores.begin(),std::min_element(scores.begin(),scores.end()));
}

bool MinusDarwin::Solver::useOpenCL() {
    return false;
}

bool MinusDarwin::Solver::checkEpsilonReached(const std::vector<float> &scores) {
    if(sParams.goal == GoalFunction::EpsilonReached) {
        return scores.at(getBestAgentId(scores)) < sParams.epsilon;
    }
    return false;
}

void MinusDarwin::Solver::crossoverPopulation(const MinusDarwin::Population &src, MinusDarwin::Population &dst, const size_t bestAgentId) {
    auto neighbours = Neighbours(sParams.popSize, std::vector<size_t>(kNeighsPerAgent,0));
    createNeighbours(neighbours,bestAgentId);

}

void MinusDarwin::Solver::selectionPopulation(const MinusDarwin::Population &a, const MinusDarwin::Population &b) {

}

void MinusDarwin::Solver::createNeighbours(MinusDarwin::Neighbours &n, const size_t bestAgentId) {
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<size_t> uid(0,sParams.popSize-1);
    for(size_t agentId = 0; agentId < sParams.popSize; agentId++) {
        auto &a = n.at(agentId);
        size_t p = 0;
        if(sParams.mode==CrossoverMode::Best) {
            a.at(p++) = bestAgentId;
        }
        for(;p<kNeighsPerAgent;p++) {
            size_t selected;
            do {
                selected = uid(re);
            } while (selected == agentId ||
                    std::find(a.begin(),a.begin()+p,selected) !=
                    a.begin()+p);
            a.at(p) = selected;
        }
    }
}

