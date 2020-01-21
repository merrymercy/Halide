#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <stdlib.h>

#include "halide_benchmark.h"


inline std::vector<int> GetArgsFromEnv() {
    std::vector<int> ret;

    if (const char* env_p = std::getenv("HL_APP_ARGS")) {
        std::string val(env_p);

        size_t offset = 0;
        auto pos = val.find(',', offset);
        while (pos != std::string::npos) {
            ret.push_back(std::stoi(val.substr(offset, pos - offset)));
            offset = pos + 1;
            pos = val.find(',', offset);
        }
        ret.push_back(std::stoi(val.substr(offset, val.size() - offset)));
    } else {
        std::cerr << "Cannot load arguments from environment variable HL_APP_ARGS" << std::endl;
        exit(-1);
    }

    return ret;
}


inline void three_way_bench(std::function<void()> manual,
                            std::function<void()> auto_classic,
                            std::function<void()> auto_new,
                            std::ostream& output = std::cout) {

    const auto is_one = [](const char *key) -> bool {
        const char *value = getenv(key);
        return value && value[0] == '1' && value[1] == 0;
    };

    Halide::Tools::BenchmarkConfig config;
    config.min_time = 1.0;
    config.max_time = 10.0;
    config.accuracy = 0.005;

    double t;

    if (manual && !is_one("HL_THREE_WAY_BENCH_SKIP_MANUAL")) {
        manual();
        t = Halide::Tools::benchmark(manual, config);
        output << "Manually-tuned time: " << t * 1e3 << " ms\n";
    }

    if (auto_classic && !is_one("HL_THREE_WAY_BENCH_SKIP_AUTO_CLASSIC")) {
        auto_classic();
        t = Halide::Tools::benchmark(auto_classic, config);
        output << "Classic auto-scheduled time: " << t * 1e3 << " ms\n";
    }

    if (auto_new && !is_one("HL_THREE_WAY_BENCH_SKIP_AUTO_NEW")) {
        auto_new();
        t = Halide::Tools::benchmark(auto_new, config);
        output << "Auto-scheduled : " << t * 1e3 << " ms\n";
    }
}
