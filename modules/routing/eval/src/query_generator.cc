#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "boost/algorithm/string.hpp"
#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/program_options.hpp"

#include "utl/erase.h"
#include "utl/to_vec.h"

#include "conf/options_parser.h"

#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;

struct generator_settings : public conf::configuration {
  generator_settings() : configuration("Generator Settings") {
    param(query_count_, "query_count", "number of queries to generate");
    param(target_file_fwd_, "target_file_fwd",
          "file(s) to write generated departure time queries to. ${target} is "
          "replaced by the target url");
    param(target_file_bwd_, "target_file_bwd",
          "file(s) to write generated arrival time queries to. ${target} is "
          "replaced by the target url");
    param(large_stations_, "large_stations",
          "use only large stations as start/destination");
    param(include_equivalent_, "include_equivalent",
          "set include_equivalent query flag");
    param(query_type_, "query_type", "query type: pretrip|ontrip_station");
    param(targets_, "targets",
          "message target urls. for every url query files will be generated");
  }

  generator_settings(generator_settings const&) = delete;
  generator_settings(generator_settings&&) = default;
  generator_settings& operator=(generator_settings const&) = delete;
  generator_settings& operator=(generator_settings&&) = default;

  ~generator_settings() override = default;

  Start get_start_type() const {
    if (query_type_ == "pretrip") {
      return Start_PretripStart;
    } else if (query_type_ == "ontrip_station") {
      return Start_OntripStationStart;
    } else {
      throw std::runtime_error{"start type not supported"};
    }
  }

  int query_count_{1000};
  std::string target_file_fwd_{"queries-fwd-${target}.txt"};
  std::string target_file_bwd_{"queries-bwd-${target}.txt"};
  bool large_stations_{false};
  bool include_equivalent_{false};
  std::string query_type_{"pretrip"};
  std::vector<std::string> targets_{"/routing"};
};

struct search_interval_generator {
  search_interval_generator(time_t begin, time_t end)
      : begin_(begin), rng_(rd_()), d_(generate_distribution(begin, end)) {
    rng_.seed(std::time(nullptr));
  }

  std::pair<time_t, time_t> random_interval() {
    auto begin = begin_ + d_(rng_) * 3600;
    return {begin, begin + 3600};
  }

private:
  static std::discrete_distribution<int> generate_distribution(time_t begin,
                                                               time_t end) {
    auto constexpr k_two_hours = 2 * 3600;
    static const int prob[] = {
        1,  // 01: 00:00 - 01:00
        1,  // 02: 01:00 - 02:00
        1,  // 03: 02:00 - 03:00
        1,  // 04: 03:00 - 04:00
        1,  // 05: 04:00 - 05:00
        2,  // 06: 05:00 - 06:00
        3,  // 07: 06:00 - 07:00
        4,  // 08: 07:00 - 08:00
        4,  // 09: 08:00 - 09:00
        3,  // 10: 09:00 - 10:00
        2,  // 11: 10:00 - 11:00
        2,  // 12: 11:00 - 12:00
        2,  // 13: 12:00 - 13:00
        2,  // 14: 13:00 - 14:00
        3,  // 15: 14:00 - 15:00
        4,  // 16: 15:00 - 16:00
        4,  // 17: 16:00 - 17:00
        4,  // 18: 17:00 - 18:00
        4,  // 19: 18:00 - 19:00
        3,  // 20: 19:00 - 20:00
        2,  // 21: 20:00 - 21:00
        1,  // 22: 21:00 - 22:00
        1,  // 23: 22:00 - 23:00
        1  // 24: 23:00 - 24:00
    };
    std::vector<int> v;
    for (time_t t = begin, hour = 0; t < end - k_two_hours; t += 3600, ++hour) {
      int h = hour % 24;
      v.push_back(prob[h]);  // NOLINT
    }
    return std::discrete_distribution<int>(std::begin(v), std::end(v));
  }

  time_t begin_;
  std::random_device rd_;
  std::mt19937 rng_;
  std::vector<int> hour_prob_;
  std::discrete_distribution<int> d_;
};

static int rand_in(int start, int end) {
  static bool initialized = false;
  static std::mt19937 rng;  // NOLINT
  if (!initialized) {
    initialized = true;
    rng.seed(std::time(nullptr));
  }

  std::uniform_int_distribution<int> dist(start, end);
  return dist(rng);
}

template <typename It>
static It rand_in(It begin, It end) {
  return std::next(begin, rand_in(0, std::distance(begin, end) - 1));
}

std::string query(std::string const& target, Start const start_type, int id,
                  std::time_t interval_start, std::time_t interval_end,
                  std::string const& from_eva, std::string const& to_eva,
                  SearchDir const dir, bool include_equivalent) {
  message_creator fbb;
  auto const interval = Interval(interval_start, interval_end);
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, start_type,
          start_type == Start_PretripStart
              ? CreatePretripStart(
                    fbb,
                    CreateInputStation(fbb, fbb.CreateString(from_eva),
                                       fbb.CreateString("")),
                    &interval)
                    .Union()
              : CreateOntripStationStart(
                    fbb,
                    CreateInputStation(fbb, fbb.CreateString(from_eva),
                                       fbb.CreateString("")),
                    interval_start)
                    .Union(),
          CreateInputStation(fbb, fbb.CreateString(to_eva),
                             fbb.CreateString("")),
          SearchType_Default, dir, fbb.CreateVector(std::vector<Offset<Via>>()),
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()), true,
          true, true, include_equivalent)
          .Union(),
      target);
  auto msg = make_msg(fbb);
  msg->get()->mutate_id(id);

  auto json = msg->to_json();
  utl::erase(json, '\n');
  return json;
}

bool has_events(edge const& e, motis::time from, motis::time to) {
  auto con = e.get_connection(from);
  return con != nullptr && con->d_time_ <= to;
}

bool has_events(station_node const& s, motis::time from, motis::time to) {
  auto found = false;
  s.for_each_route_node([&](node const* r) {
    for (auto const& e : r->edges_) {
      if (!e.empty() && has_events(e, from, to)) {
        found = true;
      }
    }
  });
  return found;
}

int random_station_id(std::vector<station_node const*> const& station_nodes,
                      time_t motis_interval_start, time_t motis_interval_end) {
  auto first = std::next(begin(station_nodes), 2);
  auto last = end(station_nodes);

  station_node const* s = nullptr;
  do {
    s = *rand_in(first, last);
  } while (!has_events(*s, motis_interval_start, motis_interval_end));
  return s->id_;
}

bool is_meta(station const* a, station const* b) {
  return std::find(begin(a->equivalent_), end(a->equivalent_), b) !=
         end(a->equivalent_);
}

std::pair<std::string, std::string> random_station_ids(
    schedule const& sched,
    std::vector<station_node const*> const& station_nodes,
    time_t interval_start, time_t interval_end) {
  station const *from = nullptr, *to = nullptr;
  auto motis_interval_start = unix_to_motistime(sched, interval_start);
  auto motis_interval_end = unix_to_motistime(sched, interval_end);
  if (motis_interval_start == INVALID_TIME ||
      motis_interval_end == INVALID_TIME) {
    std::cout << "ERROR: generated timestamp not valid:\n";
    std::cout << "  schedule range: " << sched.schedule_begin_ << " - "
              << sched.schedule_end_ << "\n";
    std::cout << "  interval_start = " << interval_start << " ("
              << format_time(motis_interval_start) << ")\n";
    std::cout << "  interval_end = " << interval_end << " ("
              << format_time(motis_interval_end) << ")\n";
    std::terminate();
  }
  do {
    from = sched.stations_
               .at(random_station_id(station_nodes, motis_interval_start,
                                     motis_interval_end))
               .get();
    to = sched.stations_
             .at(random_station_id(station_nodes, motis_interval_start,
                                   motis_interval_end))
             .get();
  } while (from == to || is_meta(from, to));
  return {from->eva_nr_.str(), to->eva_nr_.str()};
}

std::string replace_target_escape(std::string const& str,
                                  std::string const& target) {
  std::string const target_escape = "${target}";
  auto const esc_pos = str.find(target_escape);
  if (esc_pos == std::string::npos) {
    return str;
  }

  auto clean_target = target;
  if (clean_target[0] == '/') {
    clean_target.erase(clean_target.begin());
  }
  std::replace(clean_target.begin(), clean_target.end(), '/', '_');

  auto target_str = str;
  target_str.replace(esc_pos, target_escape.size(), clean_target);

  return target_str;
}

int main(int argc, char const** argv) {
  generator_settings generator_opt;
  dataset_settings dataset_opt;
  dataset_opt.adjust_footpaths_ = true;

  conf::options_parser parser({&dataset_opt, &generator_opt});
  parser.read_command_line_args(argc, argv);

  if (parser.help()) {
    std::cout << "\n\tQuery Generator\n\n";
    parser.print_help(std::cout);
    return 0;
  } else if (parser.version()) {
    std::cout << "Query Generator\n";
    return 0;
  }

  parser.read_configuration_file();

  std::cout << "\n\tQuery Generator\n\n";
  parser.print_unrecognized(std::cout);
  parser.print_used(std::cout);

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt,
                  import_settings({dataset_opt.dataset_}));

  auto const& sched = instance.sched();
  search_interval_generator interval_gen(
      sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60,
      sched.schedule_end_);

  std::vector<station_node const*> station_nodes;
  if (generator_opt.large_stations_) {
    auto const num_stations = 1000;

    auto stations = utl::to_vec(sched.stations_,
                                [](station_ptr const& s) { return s.get(); });

    std::vector<double> sizes(stations.size());
    for (auto i = 0UL; i < stations.size(); ++i) {
      auto& factor = sizes[i];
      auto const& events = stations[i]->dep_class_events_;
      for (auto i = 0UL; i < events.size(); ++i) {
        factor += std::pow(10, (9 - i) / 3) * events.at(i);
      }
    }

    std::sort(begin(stations), end(stations),
              [&](station const* a, station const* b) {
                return sizes[a->index_] > sizes[b->index_];
              });

    auto const n = std::min(static_cast<size_t>(num_stations), stations.size());
    for (auto i = 0U; i < n; ++i) {
      station_nodes.push_back(
          sched.station_nodes_.at(stations[i]->index_).get());
    }
  } else {
    station_nodes =
        utl::to_vec(sched.station_nodes_, [](station_node_ptr const& s) {
          return static_cast<station_node const*>(s.get());
        });
  }

  std::vector<std::ofstream> fwd_ofstreams;
  std::vector<std::ofstream> bwd_ofstreams;

  for (auto const& target : generator_opt.targets_) {
    auto const& fwd_fn =
        replace_target_escape(generator_opt.target_file_fwd_, target);
    auto const& bwd_fn =
        replace_target_escape(generator_opt.target_file_bwd_, target);

    fwd_ofstreams.emplace_back(fwd_fn);
    bwd_ofstreams.emplace_back(bwd_fn);
  }

  auto const start_type = generator_opt.get_start_type();
  for (int i = 1; i <= generator_opt.query_count_; ++i) {
    auto interval = interval_gen.random_interval();
    auto evas = random_station_ids(sched, station_nodes, interval.first,
                                   interval.second);

    for (auto f_idx = 0; f_idx < generator_opt.targets_.size(); ++f_idx) {
      auto const& target = generator_opt.targets_[f_idx];
      auto& out_fwd = fwd_ofstreams[f_idx];
      auto& out_bwd = bwd_ofstreams[f_idx];

      out_fwd << query(target, start_type, i, interval.first, interval.second,
                       evas.first, evas.second, SearchDir_Forward,
                       generator_opt.include_equivalent_)
              << "\n";
      out_bwd << query(target, start_type, i, interval.first, interval.second,
                       evas.first, evas.second, SearchDir_Backward,
                       generator_opt.include_equivalent_)
              << "\n";
    }
  }

  for (auto& ofstream : fwd_ofstreams) {
    ofstream.flush();
  }

  for (auto& ofstream : bwd_ofstreams) {
    ofstream.flush();
  }

  return 0;
}
