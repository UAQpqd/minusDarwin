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

void MinusDarwin::Solver::crossoverPopulation(const MinusDarwin::Population &src, MinusDarwin::Population &dst,
                                              const size_t bestAgentId) {
    auto neighbours = Neighbours(sParams.popSize, std::vector<size_t>(kNeighsPerAgent(sParams.modeDepth),0));
    createNeighbours(neighbours,bestAgentId);
    std::vector<float> coProbs(
            sParams.popSize*kNeighsPerAgent(sParams.modeDepth),0.0f);
    std::vector<float> deltas(sParams.popSize);
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_real_distribution<float> urd(0.0f,1.0f);
    std::uniform_int_distribution<size_t> uid(0,sParams.dims-1);
    std::generate(coProbs.begin(),coProbs.end(),[&urd,&re] () { return urd(re); });
    std::generate(deltas.begin(),deltas.end(),[&uid,&re] () { return uid(re); });
    for(size_t agentId = 0; agentId < sParams.popSize; agentId++) {
        auto &agentSrc = src.at(agentId);
        auto &agentDst = dst.at(agentId);
        auto &agentNgbs = neighbours.at(agentId);
        auto agentCoProbs = std::vector<float>(
                coProbs.begin()+agentId*kNeighsPerAgent(sParams.modeDepth),
                coProbs.begin()+(agentId+1)*kNeighsPerAgent(sParams.modeDepth)
        );
        const size_t delta = deltas.at(agentId);
        for(size_t paramId = 0; paramId < sParams.dims; paramId++) {
            if(agentCoProbs.at(paramId)<sParams.coProb && delta != paramId) {
                float newParam = src.at(agentNgbs.at(0)).at(paramId);
                for(size_t n = 1;n<kNeighsPerAgent(sParams.modeDepth);n++) {
                    if(n%2==1)
                        newParam += sParams.diffFactor*src.at(agentNgbs.at(n)).at(paramId);
                    else
                        newParam -= sParams.diffFactor*src.at(agentNgbs.at(n)).at(paramId);
                }
                newParam = std::min(1.0f,std::max(0.0f,newParam));
                agentDst.at(paramId) = newParam;
            }
            else {
                agentDst.at(paramId) = agentSrc.at(paramId);
            }
        }
    }
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
        for(;p<kNeighsPerAgent(sParams.modeDepth);p++) {
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

