#include "motis/paxmon/tools/generator/query_generator.h"

#include <algorithm>
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

int random_station_id(std::vector<station_node const*> const& station_nodes,
                      motis::time motis_interval_start,
                      motis::time motis_interval_end) {
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
    std::time_t interval_start, std::time_t interval_end) {
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

msg_ptr make_routing_request(std::string const& from_eva,
                             std::string const& to_eva,
                             std::time_t const interval_start,
                             std::time_t const interval_end,
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

query_generator::query_generator(const schedule& sched)
    : sched_(sched),
      interval_gen_(sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60,
                    sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60 +
                        SECONDS_A_DAY) {
  station_nodes_ =
      utl::to_vec(sched.station_nodes_, [](station_node_ptr const& s) {
        return static_cast<station_node const*>(s.get());
      });
}

motis::module::msg_ptr query_generator::get_routing_request(
    const std::string& target, Start const start_type, SearchDir const dir) {
  auto const interval = interval_gen_.random_interval();
  auto const evas = random_station_ids(sched_, station_nodes_, interval.first,
                                       interval.second);
  return make_routing_request(evas.first, evas.second, interval.first,
                              interval.second, target, start_type, dir);
}

}  // namespace motis::paxmon::tools::generator
