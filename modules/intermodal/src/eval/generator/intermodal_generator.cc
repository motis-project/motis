#include "motis/intermodal/eval/commands.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

#include "boost/filesystem.hpp"
#include "boost/math/constants/constants.hpp"

#include "utl/erase.h"
#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "utl/parser/file.h"

#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "version.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

#include "motis/intermodal/eval/bounds.h"
#include "motis/intermodal/eval/parse_bbox.h"
#include "motis/intermodal/eval/parse_poly.h"

using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::ppr;
using namespace flatbuffers;

namespace motis::intermodal::eval {

struct generator_settings : public conf::configuration {
  generator_settings() : configuration("Generator Options", "") {
    param(query_count_, "query_count", "number of queries to generate");
    param(target_file_fwd_, "target_file_fwd",
          "file to write generated queries to");
    param(target_file_bwd_, "target_file_bwd",
          "file to write generated queries to");
    param(bbox_, "bbox", "bounding box for locations");
    param(poly_file_, "poly", "bounding polygon for locations");
    param(max_walk_duration_, "max_walk_duration",
          "max. walk duration (minutes)");
    param(walk_radius_, "walk_radius",
          "radius around stations for start/destination points (meters)");
    param(large_stations_, "large_stations", "use only large stations");
  }

  int query_count_{1000};
  std::string target_file_fwd_{"queries_fwd.txt"};
  std::string target_file_bwd_{"queries_bwd.txt"};
  std::string bbox_;
  std::string poly_file_;
  int max_walk_duration_{15};
  int walk_radius_{500};
  bool large_stations_{false};
};

std::unique_ptr<bounds> parse_bounds(generator_settings const& opt) {
  if (!opt.bbox_.empty()) {
    return parse_bbox(opt.bbox_);
  } else if (!opt.poly_file_.empty()) {
    return parse_poly(opt.poly_file_);
  } else {
    return nullptr;
  }
}

struct search_interval_generator {
  search_interval_generator(unixtime begin, unixtime end)
      : begin_(begin), rng_(rd_()), d_(generate_distribution(begin, end)) {
    rng_.seed(std::time(nullptr));
  }

  std::pair<unixtime, unixtime> random_interval() {
    auto begin = begin_ + d_(rng_) * 3600;
    return {begin, begin + 3600 * 2};
  }

private:
  static std::discrete_distribution<int> generate_distribution(unixtime begin,
                                                               unixtime end) {
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
    for (unixtime t = begin, hour = 0; t < end - k_two_hours;
         t += 3600, ++hour) {
      int h = hour % 24;
      v.push_back(prob[h]);  // NOLINT
    }
    return {std::begin(v), std::end(v)};
  }

  unixtime begin_;
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

constexpr auto EQUATOR_EARTH_RADIUS = 6378137.0;

inline double scale_factor(geo::merc_xy const& mc) {
  auto const lat_rad =
      2.0 * std::atan(std::exp(mc.y_ / EQUATOR_EARTH_RADIUS)) - (geo::kPI / 2);
  return std::cos(lat_rad);
}

std::string query(int id, unixtime interval_begin, unixtime interval_end,
                  geo::latlng start_pos, geo::latlng dest_pos,
                  search_dir dir = search_dir::FWD,
                  int max_walk_duration = 15 * 60) {
  message_creator fbb;
  auto const start = Position(start_pos.lat_, start_pos.lng_);
  auto const interval = Interval(interval_begin, interval_end);
  std::vector<Offset<ModeWrapper>> modes{CreateModeWrapper(
      fbb, Mode_FootPPR,
      CreateFootPPR(fbb, CreateSearchOptions(fbb, fbb.CreateString("default"),
                                             max_walk_duration))
          .Union())};

  fbb.create_and_finish(
      MsgContent_IntermodalRoutingRequest,
      CreateIntermodalRoutingRequest(
          fbb, IntermodalStart_IntermodalPretripStart,
          CreateIntermodalPretripStart(fbb, &start, &interval, 0, false, false)
              .Union(),
          fbb.CreateVector(modes), IntermodalDestination_InputPosition,
          CreateInputPosition(fbb, dest_pos.lat_, dest_pos.lng_).Union(),
          fbb.CreateVector(modes), SearchType_Default,
          dir == search_dir::FWD ? SearchDir_Forward : SearchDir_Backward)
          .Union(),
      "/intermodal");

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
                      unixtime motis_interval_start,
                      unixtime motis_interval_end) {
  auto first = std::next(begin(station_nodes), 2);
  auto last = end(station_nodes);

  station_node const* s = nullptr;
  do {
    s = *rand_in(first, last);
  } while (!has_events(*s, motis_interval_start, motis_interval_end));
  return s->id_;
}

std::pair<station const*, station const*> random_stations(
    schedule const& sched,
    std::vector<station_node const*> const& station_nodes,
    unixtime interval_start, unixtime interval_end) {
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
    from = sched
               .stations_[random_station_id(station_nodes, motis_interval_start,
                                            motis_interval_end)]
               .get();
    to = sched
             .stations_[random_station_id(station_nodes, motis_interval_start,
                                          motis_interval_end)]
             .get();
  } while (from == to);
  return std::make_pair(from, to);
}

struct point_generator {
  explicit point_generator(bounds* bounds) : bounds_(bounds) {}

  geo::latlng random_point_near(geo::latlng const& ref, double max_dist) {
    auto const ref_merc = geo::latlng_to_merc(ref);
    geo::latlng pt;
    do {
      // http://mathworld.wolfram.com/DiskPointPicking.html
      double radius =
          std::sqrt(real_dist_(mt_)) * (max_dist / scale_factor(ref_merc));
      double angle = real_dist_(mt_) * 2 * boost::math::constants::pi<double>();
      auto const pt_merc = geo::merc_xy{ref_merc.x_ + radius * std::cos(angle),
                                        ref_merc.y_ + radius * std::sin(angle)};
      pt = geo::merc_to_latlng(pt_merc);
    } while (bounds_ != nullptr && !bounds_->contains(pt));
    return pt;
  }

private:
  std::mt19937 mt_;
  std::uniform_real_distribution<double> real_dist_;
  bounds* bounds_;
};

int generate(int argc, char const** argv) {
  using namespace motis::intermodal;
  using namespace motis::intermodal::eval;
  dataset_settings dataset_opt;
  generator_settings generator_opt;

  try {
    conf::options_parser parser({&dataset_opt, &generator_opt});
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      std::cout << "\n\tmotis-intermodal-generator (MOTIS v" << short_version()
                << ")\n\n";
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      std::cout << "motis-intermodal-generator (MOTIS v" << long_version()
                << ")\n";
      return 0;
    }

    parser.read_configuration_file(true);
    parser.print_used(std::cout);
    parser.print_unrecognized(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  auto bds = parse_bounds(generator_opt);

  auto const max_walk_duration = generator_opt.max_walk_duration_ * 60;
  auto const radius = static_cast<double>(generator_opt.walk_radius_);

  std::ofstream out_fwd(generator_opt.target_file_fwd_);
  std::ofstream out_bwd(generator_opt.target_file_bwd_);

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt,
                  import_settings({dataset_opt.dataset_}));

  auto const& sched = instance.sched();
  search_interval_generator interval_gen(
      sched.schedule_begin_ + SCHEDULE_OFFSET_MINUTES * 60,
      sched.schedule_end_ - 5 * 60 * 60);
  point_generator point_gen(bds.get());

  std::vector<station_node const*> station_nodes;
  if (generator_opt.large_stations_) {
    auto const num_stations = 1000;

    auto stations = utl::to_vec(sched.stations_,
                                [](station_ptr const& s) { return s.get(); });

    if (bds != nullptr) {
      utl::erase_if(stations, [&](station const* s) {
        return !bds->contains({s->lat(), s->lng()});
      });
    }

    std::vector<double> sizes(stations.size());
    for (auto i = 0U; i < stations.size(); ++i) {
      auto& factor = sizes[i];
      auto const& events = stations[i]->dep_class_events_;
      for (unsigned j = 0; i < events.size(); ++j) {
        factor += std::pow(10, (9 - j) / 3) * events.at(j);
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

    if (bds != nullptr) {
      utl::erase_if(station_nodes, [&](station_node const* sn) {
        auto const s = sched.stations_[sn->id_].get();
        return !bds->contains({s->lat(), s->lng()});
      });
    }
  }

  for (int i = 1; i <= generator_opt.query_count_; ++i) {
    if ((i % 100) == 0) {
      std::cout << i << "/" << generator_opt.query_count_ << "..." << std::endl;
    }
    auto const [interval_start, interval_end] =  // NOLINT
        interval_gen.random_interval();

    auto const [from, to] =  // NOLINT
        random_stations(sched, station_nodes, interval_start, interval_end);

    auto const start_pt =
        point_gen.random_point_near({from->lat(), from->lng()}, radius);
    auto const dest_pt =
        point_gen.random_point_near({to->lat(), to->lng()}, radius);

    out_fwd << query(i, interval_start, interval_end, start_pt, dest_pt,
                     search_dir::FWD, max_walk_duration)
            << "\n";
    out_bwd << query(i, interval_start, interval_end, start_pt, dest_pt,
                     search_dir::BWD, max_walk_duration)
            << "\n";
  }

  return 0;
}

}  // namespace motis::intermodal::eval
