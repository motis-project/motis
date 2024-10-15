#include "motis/paxmon/tools/generator/query_generator.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <random>

#include "utl/to_vec.h"

#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;

namespace motis::paxmon::tools::generator {

static int rand_in(int start, int end) {
  static std::mt19937 rng{std::random_device{}()};
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

station const* random_station(
    schedule const& sched,
    std::vector<station_node const*> const& station_nodes,
    motis::time motis_interval_start, motis::time motis_interval_end,
    unsigned max_attempts = 1000U) {
  auto first = begin(station_nodes);
  auto last = end(station_nodes);

  station_node const* sn = nullptr;
  for (auto attempt = 0U; attempt < max_attempts; ++attempt) {
    sn = *rand_in(first, last);
    if (has_events(*sn, motis_interval_start, motis_interval_end)) {
      return sched.stations_.at(sn->id_).get();
    }
  }

  return nullptr;
}

bool is_meta(station const* a, station const* b) {
  return std::find(begin(a->equivalent_), end(a->equivalent_), b) !=
         end(a->equivalent_);
}

std::optional<std::pair<std::string, std::string>> random_station_ids(
    schedule const& sched,
    std::vector<station_node const*> const& station_nodes,
    unixtime const interval_start, unixtime const interval_end) {
  station const *from = nullptr, *to = nullptr;
  auto const motis_interval_start = unix_to_motistime(sched, interval_start);
  auto const motis_interval_end = unix_to_motistime(sched, interval_end);
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
    from = random_station(sched, station_nodes, motis_interval_start,
                          motis_interval_end);
    if (from == nullptr) {
      return {};
    }
    to = random_station(sched, station_nodes, motis_interval_start,
                        motis_interval_end);
    if (to == nullptr) {
      return {};
    }
  } while (from == to || is_meta(from, to));
  return {{from->eva_nr_.str(), to->eva_nr_.str()}};
}

msg_ptr make_routing_request(std::string const& from_eva,
                             std::string const& to_eva,
                             unixtime const interval_start,
                             unixtime const interval_end,
                             std::string const& target, Start const start_type,
                             SearchDir const dir) {
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
          fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()))
          .Union(),
      target);
  return make_msg(fbb);
}

query_generator::query_generator(const schedule& sched,
                                 unsigned const largest_stations)
    : sched_(sched),
      interval_gen_(sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60,
                    sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60 +
                        SECONDS_A_DAY) {
  if (largest_stations > 0) {
    auto stations = utl::to_vec(sched.stations_,
                                [](station_ptr const& s) { return s.get(); });

    std::vector<double> sizes(stations.size());
    for (auto i = 0UL; i < stations.size(); ++i) {
      auto& factor = sizes[i];
      auto const& events = stations[i]->dep_class_events_;
      for (auto j = 0UL; j < events.size(); ++j) {
        factor +=
            std::pow(2.0, static_cast<double>(service_class::NUM_CLASSES) - j) *
            events.at(j);
      }
    }

    std::sort(begin(stations), end(stations),
              [&](station const* a, station const* b) {
                return sizes[a->index_] > sizes[b->index_];
              });

    auto const n =
        std::min(static_cast<size_t>(largest_stations), stations.size());
    for (auto i = 0U; i < n; ++i) {
      station_nodes_.push_back(
          sched.station_nodes_.at(stations[i]->index_).get());
    }
  } else {
    station_nodes_ =
        utl::to_vec(sched.station_nodes_, [](station_node_ptr const& s) {
          return static_cast<station_node const*>(s.get());
        });
  }
}

motis::module::msg_ptr query_generator::get_routing_request(
    const std::string& target, Start const start_type, SearchDir const dir) {
  while (true) {
    auto const interval = interval_gen_.random_interval();
    auto const stations = random_station_ids(sched_, station_nodes_,
                                             interval.first, interval.second);
    if (stations) {
      auto const& evas = stations.value();
      return make_routing_request(evas.first, evas.second, interval.first,
                                  interval.second, target, start_type, dir);
    }
  }
}

}  // namespace motis::paxmon::tools::generator
