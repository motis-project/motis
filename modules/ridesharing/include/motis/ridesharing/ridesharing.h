#pragma once

#include "boost/program_options.hpp"

#include "motis/core/common/logging.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/connection_lookup.h"
#include "motis/ridesharing/database.h"
#include "motis/ridesharing/lift.h"
#include "motis/ridesharing/query.h"
#include "motis/ridesharing/query_response.h"
#include "motis/ridesharing/routing_result.h"
#include "motis/ridesharing/statistics.h"

#include <functional>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "geo/latlng.h"

namespace po = boost::program_options;
using namespace flatbuffers;
using namespace motis::module;

namespace motis::ridesharing {

struct database;

struct ridesharing : public motis::module::module {
  ridesharing();
  ~ridesharing() override;

  ridesharing(ridesharing const&) = delete;
  ridesharing& operator=(ridesharing const&) = delete;

  ridesharing(ridesharing&&) = delete;
  ridesharing& operator=(ridesharing&&) = delete;

  void init(motis::module::registry&) override;
  rs_statistics stats_;

private:
  motis::module::msg_ptr init_module(motis::module::msg_ptr const&);
  motis::module::msg_ptr edges(motis::module::msg_ptr const&);
  motis::module::msg_ptr create(motis::module::msg_ptr const&);
  motis::module::msg_ptr remove(motis::module::msg_ptr const&);
  motis::module::msg_ptr book(motis::module::msg_ptr const&);
  motis::module::msg_ptr unbook(motis::module::msg_ptr const&);
  motis::module::msg_ptr time_out(motis::module::msg_ptr const& msg);
  motis::module::msg_ptr statistics(motis::module::msg_ptr const& msg);

  void load_lifts_from_db();
  void initialize_routing_matrix();
  // lift from_db(auto const& db_lift);

  std::pair<geo::latlng, int> add_parking(geo::latlng const&);

  std::string database_path_{":memory:"};
  size_t db_max_size_{sizeof(void*) >= 8 ? 1024ULL * 1024 * 1024 * 1024
                                         : 256 * 1024 * 1024};

  bool same_stations_;
  int close_station_radius_{30000};
  int max_stations_{10000};
  bool use_parking_{true};
  bool init_;

  std::unique_ptr<database> database_;
  std::vector<geo::latlng> station_locations_;
  std::vector<std::pair<geo::latlng, int>> parkings_;
  std::vector<std::string> station_evas_;
  std::unordered_map<std::string, int> lookup_station_evas_;
  std::map<lift_key, connection_lookup> lift_connections_;
  std::vector<std::vector<routing_result>> routing_matrix_;
};

}  // namespace motis::ridesharing
