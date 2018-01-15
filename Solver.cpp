//
// Created by dev on 1/13/2018.
//

#include "Solver.hpp"

MinusDarwin::Solver::Solver(const MinusDarwin::SolverParameterSet &t_sParams,
                            const std::vector<float> *t_data,
                            const std::string &t_scoreKernelSource,
                            bc::device t_device = bc::system::default_device()
) :
        sParams(t_sParams), dData(nullptr), device(t_device) {
    ctx = bc::context(device);
    queue = bc::command_queue(ctx, device);
    bc::program program;
    program.build_with_source(t_scoreKernelSource, ctx);
    scoreFunction = program.create_kernel("scoreFunction");
    dData = new bc::vector<float>(t_data->begin(), t_data->end(), queue);
}

void MinusDarwin::Solver::evaluatePopulation(DScores &scores, DPopulation &p) {
    scoreFunction.set_args(
            *dData,
            dData->size(),
            p,
            scores,
            sParams.popSize,
            sParams.dims
    );
    queue.enqueue_1d_range_kernel(scoreFunction, 0, sParams.popSize, 0);
}

void MinusDarwin::Solver::initPopulation(MinusDarwin::Population &p) {
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_real_distribution<float> urd(0.0f, 1.0f);
    for (auto &v : p) v = urd(re);
}

MinusDarwin::Agent MinusDarwin::Solver::run(bool verbose) {
    Population X(sParams.popSize * sParams.dims, 0.0f);
    Population Y(sParams.popSize * sParams.dims, 0.0f);
    Scores Xscores(sParams.popSize);
    DPopulation dX(X.begin(), X.end(), queue);
    DScores dXscores(Xscores.begin(), Xscores.end());
    DPopulation dY(sParams.popSize * sParams.dims, ctx);
    DScores dYscores(sParams.popSize, ctx);
    initPopulation(X);
    evaluatePopulation(dXscores, dX);
    size_t bestAgentId = getBestAgentId(Xscores);
    bool goalReached = checkEpsilonReached(dXscores,bestAgentId);
    if (verbose) {
        bc::copy(dX.begin(), dX.end(), X, queue);
        showPopulationHead(X, Xscores, 10);
    }
    for (size_t g = 0; g < sParams.maxGens && !goalReached; g++) {
        crossoverPopulation(dX, dY, sParams.mode == CrossoverMode::Best ? getBestAgentId(dXscores) : 0);
        evaluatePopulation(dYscores, dY);
        selectionPopulation(X, Xscores, Y, Yscores);
        goalReached = checkEpsilonReached(Xscores);
        if (verbose)
            showPopulationHead(X, Xscores, 10);
    }
    if (verbose) {
        std::cout << "Best Agent: " << X.at(bestAgentId)
                  << " " << Xscores.at(bestAgentId) << std::endl;
    }
    Agent bestAgent(sParams.dims,0.0f);
    bc::copy(
            dX.begin()+bestAgentId*sParams.dims,
            dX.begin()+(bestAgentId+1)*sParams.dims,queue);
    return bestAgent;
}

size_t MinusDarwin::Solver::getBestAgentId(const std::vector<float> &scores) {
    return std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
}

size_t MinusDarwin::Solver::getBestAgentId(const MinusDarwin::DScores &scores) {
    auto minIt = bc::min_element(scores.begin(), scores.end(), queue);
    return minIt.get_index();
}

bool MinusDarwin::Solver::checkEpsilonReached(const std::vector<float> &scores, const size_t bestAgentId) {
    if (sParams.goal == GoalFunction::EpsilonReached) {
        return scores.at(bestAgentId) < sParams.epsilon;
    }
    return false;
}

bool MinusDarwin::Solver::checkEpsilonReached(const MinusDarwin::DScores &scores, const size_t bestAgentId) {
    if (sParams.goal == GoalFunction::EpsilonReached) {
        return scores.at(bestAgentId) < sParams.epsilon;
    }
    return false;
}

void MinusDarwin::Solver::crossoverPopulation(const MinusDarwin::Population &src, MinusDarwin::Population &dst,
                                              const size_t bestAgentId) {
    auto neighbours = Neighbours(sParams.popSize, std::vector<size_t>(kNeighsPerAgent(sParams.modeDepth), 0));
    createNeighbours(neighbours, bestAgentId);
    std::vector<float> coProbs(
            sParams.popSize * kNeighsPerAgent(sParams.modeDepth), 0.0f);
    std::vector<float> deltas(sParams.popSize);
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_real_distribution<float> urd(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> uid(0, sParams.dims - 1);
    std::generate(coProbs.begin(), coProbs.end(), [&urd, &re]() { return urd(re); });
    std::generate(deltas.begin(), deltas.end(), [&uid, &re]() { return uid(re); });
    for (size_t agentId = 0; agentId < sParams.popSize; agentId++) {
        auto &agentSrc = src.at(agentId);
        auto &agentDst = dst.at(agentId);
        auto &agentNgbs = neighbours.at(agentId);
        auto agentCoProbs = std::vector<float>(
                coProbs.begin() + agentId * kNeighsPerAgent(sParams.modeDepth),
                coProbs.begin() + (agentId + 1) * kNeighsPerAgent(sParams.modeDepth)
        );
        const size_t delta = deltas.at(agentId);
        for (size_t paramId = 0; paramId < sParams.dims; paramId++) {
            if (agentCoProbs.at(paramId) < sParams.coProb && delta != paramId) {
                float newParam = src.at(agentNgbs.at(0)).at(paramId);
                for (size_t n = 1; n < kNeighsPerAgent(sParams.modeDepth); n++) {
                    if (n % 2 == 1)
                        newParam += sParams.diffFactor * src.at(agentNgbs.at(n)).at(paramId);
                    else
                        newParam -= sParams.diffFactor * src.at(agentNgbs.at(n)).at(paramId);
                }
                newParam = std::min(1.0f, std::max(0.0f, newParam));
                agentDst.at(paramId) = newParam;
            } else {
                agentDst.at(paramId) = agentSrc.at(paramId);
            }
        }
    }
}


void MinusDarwin::Solver::createNeighbours(MinusDarwin::Neighbours &n, const size_t bestAgentId) {
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<size_t> uid(0, sParams.popSize - 1);
    for (size_t agentId = 0; agentId < sParams.popSize; agentId++) {
        auto &a = n.at(agentId);
        size_t p = 0;
        if (sParams.mode == CrossoverMode::Best) {
            a.at(p++) = bestAgentId;
        }
        for (; p < kNeighsPerAgent(sParams.modeDepth); p++) {
            size_t selected;
            do {
                selected = uid(re);
            } while (selected == agentId ||
                     std::find(a.begin(), a.begin() + p, selected) !=
                     a.begin() + p);
            a.at(p) = selected;
        }
    }
}

void MinusDarwin::Solver::selectionPopulation(MinusDarwin::Population &main, std::vector<float> &mainScores,
                                              const MinusDarwin::Population &other, std::vector<float> &otherScores) {
    for (size_t a = 0; a < sParams.popSize; a++) {
        if (otherScores.at(a) < mainScores.at(a)) {
            mainScores.at(a) = otherScores.at(a);
            main.at(a) = other.at(a);
        }
    }
}

void MinusDarwin::Solver::showPopulationHead(const MinusDarwin::Population &p, const std::vector<float> &s, size_t n) {
    for (size_t a = 0; a < n; a++) {
        std::cout << "Agent: ";
        std::cout << p.at(a);
        std::cout << " " << s.at(a) << std::endl;
    }
}


void MinusDarwin::Solver::crossoverPopulation(const MinusDarwin::DPopulation &src, MinusDarwin::DPopulation &dst,
                                              const size_t bestAgentId) {

}

