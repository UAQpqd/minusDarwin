//
// Created by dev on 1/13/2018.
//

#include "Solver.hpp"

void MinusDarwin::Solver::initPopulation(MinusDarwin::Population &p) {
    boost::random::mt19937 re(boost::chrono::high_resolution_clock::now().time_since_epoch().count());
    boost::random::uniform_real_distribution<float> urd(0.0f, 1.0f);
    for (auto &a : p) for(auto &v : a) v = urd(re);
}

MinusDarwin::Agent MinusDarwin::Solver::run(bool verbose) {
    size_t bestAgentId;
    Population X(sParams.popSize, Agent(sParams.dims,0.0f));
    std::vector<float> Xscores(sParams.popSize);
    initPopulation(X);
    Population Y(sParams.popSize, Agent(sParams.dims,0.0f));
    std::vector<float> Yscores(sParams.popSize);

    auto startTime =
            boost::chrono::high_resolution_clock::now();
    evaluatePopulation(Xscores, X);
    auto endTime =
            boost::chrono::high_resolution_clock::now();
    auto duration =
            endTime-startTime;
    tracer.generationsDuration.push_back(
            boost::chrono::duration_cast<boost::chrono::milliseconds>(duration).count()
    );
    bestAgentId = getBestAgentId(Xscores);
    bool goalReached = checkEpsilonReached(Xscores,bestAgentId);
    tracer.generations.push_back(X);
    tracer.generationsScores.push_back(Xscores);
    if (verbose)
        showPopulationHead(X, Xscores, 10);
    for (size_t g = 0; g < sParams.maxGens && !goalReached; g++) {

        startTime =
                boost::chrono::high_resolution_clock::now();

        crossoverPopulation(X, Y, sParams.mode == CrossoverMode::Best ? getBestAgentId(Xscores) : 0);
        evaluatePopulation(Yscores, Y);
        selectionPopulation(X, Xscores, Y, Yscores);
        goalReached = checkEpsilonReached(Xscores, bestAgentId);
        if (verbose)
            showPopulationHead(X, Xscores, 10);
        tracer.generations.push_back(X);
        tracer.generationsScores.push_back(Xscores);

        endTime =
                boost::chrono::high_resolution_clock::now();
        duration =
                endTime-startTime;
        tracer.generationsDuration.push_back(
            boost::chrono::duration_cast<boost::chrono::milliseconds>(duration).count()
        );
    }
    bestAgentId = getBestAgentId(Xscores);
    if (verbose) {
        std::cout << "Best Agent: " << X.at(bestAgentId)
                  << " " << Xscores.at(bestAgentId) << std::endl;
    }
    tracer.bestAgent = X.at(bestAgentId);
    tracer.bestAgentScore = Xscores.at(bestAgentId);
    return X.at(bestAgentId);
}

void MinusDarwin::Solver::createNeighbours(MinusDarwin::Neighbours &n, const size_t bestAgentId) {
    boost::random::mt19937 re(boost::chrono::high_resolution_clock::now().time_since_epoch().count());
    boost::random::uniform_int_distribution<size_t> uid(0, sParams.popSize - 1);
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

void MinusDarwin::Solver::showPopulationHead(const Population &p, const Scores &s, size_t n) {
    for (size_t a = 0; a < n; a++) {
        std::cout << "Agent " << a << ": [";
        std::cout << agentCSV(p,a);
        std::cout << "] " << s.at(a) << std::endl;
    }
}

std::string MinusDarwin::Solver::agentCSV(const MinusDarwin::Population &p, const size_t &agentId) {
    std::stringstream ss;
    auto &a = p.at(agentId);
    for (size_t k = 0; k < sParams.dims; ++k) {
        ss << a.at(k);
        if(k != sParams.dims-1) ss << ",";
    }
    return ss.str();
}