
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
using namespace motis::isochrone;

struct generator_settings : public conf::configuration {
  generator_settings() : configuration("Generator Settings") {
    param(query_count_, "query_count", "number of queries to generate");
    param(target_file_, "target_file",
          "file to write generated departure time queries to. ${target} is "
          "replaced by the target url");
  }

  int query_count_{10000};
  std::string target_file_{"queries-isochrone.txt"};
};

struct search_interval_generator {
  search_interval_generator(time_t begin, time_t end)
          : begin_(begin), rng_(rd_()), d_(generate_distribution(begin, end)) {
    rng_.seed(std::time(nullptr));
  }

  std::pair<time_t, time_t> random_interval() {
    auto begin = begin_ + d_(rng_) * 3600;
    auto end = begin + 3600;
    return {begin, end};
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

std::string query(int id,
                  std::time_t interval_start, std::time_t travel_time,
                  std::string const& from_eva) {
  message_creator fbb;
  fbb.create_and_finish(
          MsgContent_IsochroneRequest,
          CreateIsochroneRequest(
                  fbb,
                  CreateInputStation(fbb, fbb.CreateString(from_eva),
                                     fbb.CreateString("")),
                  interval_start,travel_time)
                  .Union(),
          "/isochrone");
  auto msg = make_msg(fbb);
  msg->get()->mutate_id(id);

  auto json = msg->to_json();
  utl::erase(json, '\n');
  return json;
}

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

int main(int argc, char const** argv) {
  generator_settings generator_opt;
  dataset_settings dataset_opt;
  dataset_opt.adjust_footpaths_ = true;

  conf::options_parser parser({&dataset_opt, &generator_opt});
  parser.read_command_line_args(argc, argv);

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
  station_nodes =
          utl::to_vec(sched.station_nodes_, [](station_node_ptr const& s) {
            return static_cast<station_node const*>(s.get());
          });

  std::ofstream ofs(generator_opt.target_file_);
  for (int i = 1; i <= generator_opt.query_count_; ++i) {
    auto interval = interval_gen.random_interval();
    auto evas = random_station_ids(sched, station_nodes, interval.first,
                                   interval.second);
    ofs << query( i, interval.first, interval.second-interval.first,
                     evas.first)
            << "\n";


  }
  ofs.flush();
  return 0;
}
