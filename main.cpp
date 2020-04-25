#include <bits/stdc++.h>

using std::vector, std::map, std::pair;

template<class T>
using matrix = vector<vector<T>>;

namespace netcomm_utils {

vector<double> uncertainty(size_t m) {
    return vector(m, 1.0 / m);
}

matrix<double> dialogue_matrix(double p, double q) {
    return {{p, 1 - p}, {1 - q, q}};
}

double normalized_entropy(const vector<double> &distribution) {
    double h = 0;
    for (auto p: distribution) {
        if (p > 0)
            h -= p * log2(p);
    }
    return h;
}

template<class RNG>
bool bernoulli_trial(double p, RNG &generator) {
    return std::uniform_real_distribution<double>(0, 1)(generator) < p;
}

template<class RNG>
int random_choice(vector<double> p, RNG &generator) {
    double value = std::uniform_real_distribution<double>(0, 1)(generator);
    double sum = 0;
    for (int i = 0; i < p.size(); ++i) {
        sum += p[i];
        if (value <= sum)
            return i;
    }
    assert(value <= sum);
    return p.size() - 1;
}

}

template<class IndexType, class VertexData, class EdgeData>
struct UndirectedGraph {

    UndirectedGraph(const map<IndexType, VertexData> &vertexes, const map<pair<IndexType, IndexType>, EdgeData> &edges):
        vertexes(vertexes),
        edges(edges)
    {
        // no-op
    }

    static UndirectedGraph * make_complete(const map<IndexType, VertexData> &vertexes) {
        map<pair<IndexType, IndexType>, EdgeData> edges;
        for (auto i = vertexes.begin(); i != vertexes.end(); ++i) {
            for (auto j = i; j != vertexes.end(); ++j) {
                if (j == i) continue;
                edges[{i->first, j->first}] = EdgeData();
            }
        }
        return new UndirectedGraph(vertexes, edges);
    }

    map<IndexType, VertexData> vertexes;
    map<pair<IndexType, IndexType>, EdgeData> edges;
};

template<class RNG>
class Simulation {

private:

    struct VertexData {
        double rho;
        int choice;
        vector<vector<double>> result_list;
        vector<double> preference_density;
    };

    struct EdgeData {
        double a;
        matrix<double> D;

        EdgeData& operator=(const EdgeData &rhs) {
            a = rhs.a;
            D = rhs.D;
            return *this;
        }
    };

public:

    struct ObservationResult {
        vector<double> avg_preference_density;
        double disclaimed_percentage;
        vector<double> choice_density;
    };

public:

    Simulation(const vector<vector<double>> &original_preference_densities, RNG& rng): rng(rng) {
        n = original_preference_densities.size();
        nvars = original_preference_densities[0].size();
        configure_graph(original_preference_densities);
    }

    vector<ObservationResult> run(int niter) {
        auto protocol = vector{observation()};

        for (int i = 0; i < niter; ++i) {
            simulate_session();
            protocol.push_back(observation());
        }

        return protocol;
    }

    ~Simulation() {
        delete net;
    }

    static const int DISCLAIMER = -1;

private:

    int n;
    int nvars;
    RNG &rng;
    UndirectedGraph<int, VertexData, EdgeData> *net;

    void configure_graph(const vector<vector<double>> &original_preference_densities) {

        // set the parameters of the community actors
        map<int, VertexData> vertexes;
        for (int i = 0; i < n; ++i) {
            vertexes[i] = {
                20,
                i == 0 ? 0 : DISCLAIMER,
                {},
                original_preference_densities[i]
            };
        }

        // initialize the graph
        net = UndirectedGraph<int, VertexData, EdgeData>::make_complete(vertexes);

        // set the parameters of the community channels
        std::uniform_real_distribution<double> distr(0.2, 0.6);
        for (auto &[edge, data]: net->edges) {
            auto [alice, bob] = std::minmax(edge.first, edge.second);
            if (alice ==  0) {
                data.a = 1.0;
                data.D = netcomm_utils::dialogue_matrix(1.0, distr(rng));
            } else {
                data.a = 1.0;
                data.D = netcomm_utils::dialogue_matrix(distr(rng), distr(rng));
            }
        }
    }

    auto observation() -> ObservationResult {

        // simulate polling
        int disclaimed_amount = 0;
        for (auto &[_, vertex]: net->vertexes) {
            auto hn = netcomm_utils::normalized_entropy(vertex.preference_density);
            if (netcomm_utils::bernoulli_trial(pow(hn, vertex.rho), rng)) {
                vertex.choice = DISCLAIMER;
                disclaimed_amount++;
            } else {
                vertex.choice = netcomm_utils::random_choice(vertex.preference_density, rng);
            }
        }

        // compute average preference density
        vector<double> avg_preference_density(nvars);

        for (int j = 0; j < nvars; ++j) {
            for (auto &[_, vertex]: net->vertexes) {
                avg_preference_density[j] += vertex.preference_density[j];
            }
            avg_preference_density[j] /= n;
        }

        if (disclaimed_amount == n) {
            return {avg_preference_density, 1.0, netcomm_utils::uncertainty(nvars)};
        }

        int voted_amount = n - disclaimed_amount;

        // compute polling results
        vector<double> choice_density(nvars);
        for(auto &[_, vertex]: net->vertexes) {
            if (vertex.choice != DISCLAIMER) {
                choice_density[vertex.choice] += 1;
            }
        }
        for (auto &choice: choice_density)
            choice /= voted_amount;

        double disclaimed_percentage = disclaimed_amount * 1.0 / n;

        return {avg_preference_density, disclaimed_percentage, choice_density};
    }

    auto simulate_dialogue(int alice, int bob) -> pair<vector<double>, vector<double>> {
        if (alice > bob)
            std::swap(alice, bob);

        auto D = net->edges[{alice, bob}].D;
        auto wA = net->vertexes[alice].preference_density;
        auto wB = net->vertexes[bob].preference_density;
        auto wA_result = vector<double>(nvars);
        auto wB_result = vector<double>(nvars);

        for (int i = 0; i < nvars; ++i) {
            wA_result[i] = D[0][0] * wA[i] + D[0][1] * wB[i];
            wB_result[i] = D[1][0] * wA[i] + D[1][1] * wB[i];
        }

        return {wA_result, wB_result};
    }

    void simulate_session() {
        // clear auxiliary information
        for (auto &[_, vertex]: net->vertexes)
            vertex.result_list.clear();

        // simulate sesion dialogues
        for (auto &[edge, data]: net->edges) {
            if (!netcomm_utils::bernoulli_trial(data.a, rng))
                continue;

            auto [alice, bob] = std::minmax(edge.first, edge.second);

            auto [wA_result, wB_result] = simulate_dialogue(alice, bob);
            net->vertexes[alice].result_list.push_back(wA_result);
            net->vertexes[bob].result_list.push_back(wB_result);
        }

        // compute session result for each actor
        for (auto &[_, vertex]: net->vertexes) {
            if (!vertex.result_list.empty()) {
                vertex.preference_density.assign(nvars, 0);

                for (int i = 0; i < nvars; ++i) {
                    for (auto &dialogue_result: vertex.result_list)
                        vertex.preference_density[i] += dialogue_result[i];
                    vertex.preference_density[i] /= vertex.result_list.size();
                }
            }
        }
    }
};

int main() {

    // number of actors
    int n = 200;

    // specify initial prefernce densities of community actors
    vector<vector<double>> original_preference_densities(n);

    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            original_preference_densities[i] = {1.0, 0.0};
        } else if (i == 1) {
            original_preference_densities[i] = {0.0, 1.0};
        } else {
            original_preference_densities[i] = netcomm_utils::uncertainty(2);
        }
    }

    // set up the simulation
    std::mt19937 rng;
    rng.seed(1234);
    Simulation<std::mt19937> s(original_preference_densities, rng);

    // run the simulation
    auto protocol = s.run(100);

    // save the results
    std::ofstream cout("protocol.dat");
    std::ios_base::sync_with_stdio(0);
    cout.tie(0);

    for (auto &item: protocol) {
        for (auto preference: item.avg_preference_density)
            cout << preference << " ";
        cout << item.disclaimed_percentage;
        for (auto choise: item.choice_density)
            cout << " " << choise;
        cout << "\n";
    }
}
