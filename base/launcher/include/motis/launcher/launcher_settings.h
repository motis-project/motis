#pragma once

#include <string>
#include <thread>

#include "boost/program_options.hpp"

#include "conf/configuration.h"

namespace motis::launcher {

class launcher_settings : public conf::configuration {
public:
  enum class motis_mode_t { BATCH, SERVER, TEST, INIT };

  friend std::istream& operator>>(std::istream& in, motis_mode_t& mode) {
    std::string token;
    in >> token;

    if (token == "batch") {
      mode = launcher_settings::motis_mode_t::BATCH;
    } else if (token == "server") {
      mode = launcher_settings::motis_mode_t::SERVER;
    } else if (token == "test") {
      mode = launcher_settings::motis_mode_t::TEST;
    } else if (token == "init") {
      mode = launcher_settings::motis_mode_t::INIT;
    }

    return in;
  }

  friend std::ostream& operator<<(std::ostream& out, motis_mode_t const& mode) {
    switch (mode) {
      case launcher_settings::motis_mode_t::BATCH: out << "batch"; break;
      case launcher_settings::motis_mode_t::SERVER: out << "server"; break;
      case launcher_settings::motis_mode_t::TEST: out << "test"; break;
      case launcher_settings::motis_mode_t::INIT: out << "init"; break;
    }
    return out;
  }

  launcher_settings() : conf::configuration{"Launcher Settings"} {
    param(mode_, "mode",
          "Mode of operation:\n"
          "batch = inject batch file\n"
          "server = network server\n"
          "test = exit after 1s");
    param(batch_input_file_, "batch_input_file", "query file");
    param(batch_output_file_, "batch_output_file", "response file");
    param(init_, "init", "init operation");
    param(num_threads_, "num_threads", "number of worker threads");
    param(direct_mode_, "direct", "no ctx/multi-threading");
    param(otlp_http_, "otlp_http", "enable OTLP HTTP exporter");
  }

  motis_mode_t mode_{launcher_settings::motis_mode_t::SERVER};
  std::string batch_input_file_{"queries.txt"};
  std::string batch_output_file_{"responses.txt"};
  std::string init_;
  unsigned num_threads_{std::thread::hardware_concurrency()};
  bool direct_mode_{sizeof(void*) >= 8 ? false : true};
  bool otlp_http_{};
};

}  // namespace motis::launcher
