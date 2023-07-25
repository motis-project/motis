#include "motis/intermodal/eval/commands.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "boost/math/constants/constants.hpp"

#define CISTA_PRINTABLE_NO_VEC
#include "cista/reflection/printable.h"

#include "utl/erase.h"
#include "utl/erase_if.h"
#include "utl/parser/cstr.h"
#include "utl/parser/file.h"
#include "utl/to_vec.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"
#include "motis/intermodal/eval/bounds.h"
#include "motis/intermodal/eval/parse_bbox.h"
#include "motis/intermodal/eval/parse_poly.h"

#include "utl/zip.h"
#include "version.h"

using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::ppr;
namespace fbs = flatbuffers;

namespace motis::intermodal::eval {

constexpr auto const kTargetEscape = std::string_view{"TARGET"};

struct generator_settings : public conf::configuration {
  generator_settings() : configuration("Generator Options", "") {
    param(query_count_, "query_count", "number of queries to generate");
    param(out_, "out", "file to write generated queries to");
    param(bbox_, "bbox", "bounding box for locations");
    param(poly_file_, "poly", "bounding polygon for locations");
    param(start_modes_, "start_modes", "start modes ppr-15|osrm_car-15|...");
    param(dest_modes_, "dest_modes", "destination modes (see start modes)");
    param(large_stations_, "large_stations", "use only large stations");
    param(message_type_, "message_type", "intermodal|routing");
    param(start_type_, "start_type",
          "query type:\n"
          "  pretrip = interval at station\n"
          "  ontrip_station = start time at station\n"
          "  ontrip_train = start in train\n"
          "  intermodal_pretrip = interval at coordinate\n"
          "  intermodal_ontrip = start time at station");
    param(dest_type_, "dest_type", "destination type: coordinate|station");
    param(routers_, "routers", "routing targets");
    param(search_dir_, "search_dir", "search direction forward/backward");
    param(extend_earlier_, "extend_earlier", "extend search interval earlier");
    param(extend_later_, "extend_later", "extend search interval later");
    param(min_connection_count_, "min_connection_count",
          "min. number of connections (otherwise interval will be extended)");
  }

  MsgContent get_message_type() const {
    using cista::hash;
    switch (hash(message_type_)) {
      case hash("routing"): return MsgContent_RoutingRequest;
      case hash("intermodal"): return MsgContent_IntermodalRoutingRequest;
    }
    throw std::runtime_error{"query type not "};
  }

  IntermodalStart get_start_type() const {
    using cista::hash;
    switch (hash(start_type_)) {
      case hash("intermodal_pretrip"):
        return IntermodalStart_IntermodalPretripStart;
      case hash("intermodal_ontrip"):
        return IntermodalStart_IntermodalOntripStart;
      case hash("ontrip_train"): return IntermodalStart_OntripTrainStart;
      case hash("ontrip_station"): return IntermodalStart_OntripStationStart;
      case hash("pretrip"): return IntermodalStart_PretripStart;
    }
    throw std::runtime_error{"start type not supported"};
  }

  IntermodalDestination get_dest_type() const {
    using cista::hash;
    switch (hash(dest_type_)) {
      case hash("coordinate"): return IntermodalDestination_InputPosition;
      case hash("station"): return IntermodalDestination_InputStation;
    }
    throw std::runtime_error{"start type not supported"};
  }

  SearchDir get_search_dir() const {
    using cista::hash;
    switch (hash(search_dir_)) {
      case hash("forward"): return SearchDir_Forward;
      case hash("backward"): return SearchDir_Backward;
    }
    throw std::runtime_error{"search dir not supported"};
  }

  int query_count_{1000};
  std::string message_type_{"intermodal"};
  std::string out_{"q_TARGET.txt"};
  std::string bbox_;
  std::string poly_file_;
  std::string start_modes_;
  std::string dest_modes_;
  std::string start_type_{"intermodal_pretrip"};
  std::string dest_type_{"coordinate"};
  bool large_stations_{false};
  std::vector<std::string> routers_{"/routing"};
  std::string search_dir_{"forward"};
  bool extend_earlier_{false};
  bool extend_later_{false};
  unsigned min_connection_count_{0U};
};

std::string replace_target_escape(std::string const& str,
                                  std::string const& target) {
  auto const esc_pos = str.find(kTargetEscape);
  utl::verify(esc_pos != std::string::npos, "target escape {} not found in {}",
              kTargetEscape, str);

  auto clean_target = target;
  if (clean_target[0] == '/') {
    clean_target.erase(clean_target.begin());
  }
  std::replace(clean_target.begin(), clean_target.end(), '/', '_');

  auto target_str = str;
  target_str.replace(esc_pos, kTargetEscape.size(), clean_target);

  return target_str;
}

struct mode {
  CISTA_PRINTABLE(mode);

  int get_param(std::size_t const index, int const default_value) const {
    return parameters_.size() > index ? parameters_[index] : default_value;
  }

  std::string name_;
  std::vector<int> parameters_;
};

std::vector<mode> read_modes(std::string const& in) {
  std::regex word_regex("([_a-z]+)(-\\d+)?(-\\d+)?");
  std::smatch match;
  std::vector<mode> modes;
  utl::for_each_token(utl::cstr{in}, '|', [&](auto&& s) {
    auto const x = std::string{s.view()};
    auto const matches = std::regex_search(x, match, word_regex);
    if (!matches) {
      throw utl::fail("invalid mode in \"{}\": {}", in, s.view());
    }

    auto m = mode{};
    m.name_ = match[1].str();
    if (match.size() > 2) {
      for (auto i = 2; i != match.size(); ++i) {
        auto const& group = match[i];
        if (group.str().size() > 1) {
          m.parameters_.emplace_back(std::stoi(group.str().substr(1)));
        }
      }
    }
    modes.emplace_back(m);
  });
  return modes;
}

double get_radius_in_m(std::vector<mode> const& modes) {
  constexpr auto const max_walk_speed = 0.55;  // m/s 2km/h
  constexpr auto const max_bike_speed = 3.5;  // m/s 12.5km/h
  constexpr auto const max_car_speed = 13.9;  // m/s 50km/h

  auto r = std::numeric_limits<double>::min();
  for (auto const& m : modes) {
    if (m.name_ == "ppr" || m.name_ == "osrm_foot") {
      r = std::max(r, m.get_param(0, 15) * 60 * max_walk_speed);
    } else if (m.name_ == "osrm_bike") {
      r = std::max(r, m.get_param(0, 15) * 60 * max_bike_speed);
    } else if (m.name_ == "osrm_car" || m.name_ == "osrm_car_parking") {
      r = std::max(r, m.get_param(0, 15) * 60 * max_car_speed);
    } else if (m.name_ == "gbfs") {
      r = std::max(r, m.get_param(0, 15) * 60 * max_walk_speed +
                          m.get_param(1, 15) * 60 * max_bike_speed);
    } else {
      throw utl::fail("unknown mode \"{}\"", m.name_);
    }
  }
  return r;
}

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

  Interval random_interval() {
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
      int const h = hour % 24;
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

namespace {

int rand_in(int start, int end) {
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
It rand_in(It begin, It end) {
  return std::next(begin, rand_in(0, std::distance(begin, end) - 1));
}

}  // namespace

constexpr auto EQUATOR_EARTH_RADIUS = 6378137.0;

inline double scale_factor(geo::merc_xy const& mc) {
  auto const lat_rad =
      2.0 * std::atan(std::exp(mc.y_ / EQUATOR_EARTH_RADIUS)) - (geo::kPI / 2);
  return std::cos(lat_rad);
}

std::vector<fbs::Offset<ModeWrapper>> create_modes(
    fbs::FlatBufferBuilder& fbb, std::vector<mode> const& modes) {
  auto v = std::vector<fbs::Offset<ModeWrapper>>{};
  for (auto const& m : modes) {
    if (m.name_ == "ppr") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_FootPPR,
          CreateFootPPR(fbb,
                        CreateSearchOptions(fbb, fbb.CreateString("default"),
                                            60 * m.get_param(0, 15)))
              .Union()));
    } else if (m.name_ == "osrm_foot") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_Foot, CreateFoot(fbb, 60 * m.get_param(0, 15)).Union()));
    } else if (m.name_ == "osrm_bike") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_Bike, CreateBike(fbb, 60 * m.get_param(0, 15)).Union()));
    } else if (m.name_ == "osrm_car") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_Car, CreateCar(fbb, 60 * m.get_param(0, 15)).Union()));
    } else if (m.name_ == "osrm_car_parking") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_CarParking,
          CreateCarParking(fbb, 60 * m.get_param(0, 15),
                           CreateSearchOptions(fbb, fbb.CreateString("default"),
                                               60 * m.get_param(1, 10)))
              .Union()));
    } else if (m.name_ == "gbfs") {
      v.emplace_back(CreateModeWrapper(
          fbb, Mode_GBFS,
          CreateGBFS(fbb, fbb.CreateString("default"), 60 * m.get_param(0, 15),
                     60 * m.get_param(1, 15))
              .Union()));
    } else {
      throw utl::fail("unknown mode \"{}\"", m.name_);
    }
  }
  return v;
}

struct point_generator {
  explicit point_generator(bounds* bounds) : bounds_(bounds) {}

  Position random_point_near(geo::latlng const& ref, double max_dist) {
    auto const ref_merc = geo::latlng_to_merc(ref);
    geo::latlng pt;
    do {
      // http://mathworld.wolfram.com/DiskPointPicking.html
      double const radius =
          std::sqrt(real_dist_(mt_)) * (max_dist / scale_factor(ref_merc));
      double const angle =
          real_dist_(mt_) * 2 * boost::math::constants::pi<double>();
      auto const pt_merc = geo::merc_xy{ref_merc.x_ + radius * std::cos(angle),
                                        ref_merc.y_ + radius * std::sin(angle)};
      pt = geo::merc_to_latlng(pt_merc);
    } while (bounds_ != nullptr && !bounds_->contains(pt));
    return {pt.lat_, pt.lng_};
  }

private:
  std::mt19937 mt_;
  std::uniform_real_distribution<double> real_dist_;
  bounds* bounds_;
};

std::pair<trip const*, access::trip_stop> random_trip_and_station(
    schedule const& sched) {
  auto const random_trip =
      (*rand_in(begin(sched.trips_), end(sched.trips_))).second;

  auto const stops = motis::access::stops(random_trip);
  auto const random_stop = *rand_in(stops.begin() + 1, stops.end());

  return std::pair{cista::ptr_cast(random_trip), random_stop};
}

void write_query(schedule const& sched, point_generator& point_gen, int id,
                 Interval const interval, station const* from,
                 station const* to, std::vector<mode> const& start_modes,
                 std::vector<mode> const& dest_modes, double const start_radius,
                 double const dest_radius, MsgContent const message_type,
                 IntermodalStart const start_type,
                 IntermodalDestination const destination_type, SearchDir dir,
                 bool const extend_earlier, bool const extend_later,
                 unsigned const min_connection_count,
                 std::vector<std::string> const& routers,
                 std::vector<std::ofstream>& out_files) {
  auto fbbs = utl::to_vec(
      routers, [](auto&&) { return std::make_unique<message_creator>(); });

  auto const dest_pt =
      point_gen.random_point_near({to->lat(), to->lng()}, dest_radius);

  auto const get_destination = [&](fbs::FlatBufferBuilder& fbb) {
    return destination_type == IntermodalDestination_InputPosition
               ? CreateInputPosition(fbb, dest_pt.lat(), dest_pt.lng()).Union()
               : CreateInputStation(fbb, fbb.CreateString(to->eva_nr_),
                                    fbb.CreateString(""))
                     .Union();
  };

  if (message_type == MsgContent_IntermodalRoutingRequest) {
    switch (start_type) {
      case IntermodalStart_IntermodalPretripStart: {
        auto const start_pt = point_gen.random_point_near(
            {from->lat(), from->lng()}, start_radius);

        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_IntermodalRoutingRequest,
              CreateIntermodalRoutingRequest(
                  fbb, start_type,
                  CreateIntermodalPretripStart(fbb, &start_pt, &interval,
                                               min_connection_count,
                                               extend_earlier, extend_later)
                      .Union(),
                  fbb.CreateVector(create_modes(fbb, start_modes)),
                  destination_type, get_destination(fbb),
                  fbb.CreateVector(create_modes(fbb, dest_modes)),
                  SearchType_Default, dir, fbb.CreateString(router))
                  .Union(),
              "/intermodal", DestinationType_Module, id);
        }

        break;
      }

      case IntermodalStart_IntermodalOntripStart: {
        auto const start_pt = point_gen.random_point_near(
            {from->lat(), from->lng()}, start_radius);

        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_IntermodalRoutingRequest,
              CreateIntermodalRoutingRequest(
                  fbb, start_type,
                  CreateIntermodalOntripStart(fbb, &start_pt, interval.begin())
                      .Union(),
                  fbb.CreateVector(create_modes(fbb, start_modes)),
                  destination_type,
                  destination_type == IntermodalDestination_InputPosition
                      ? CreateInputPosition(fbb, dest_pt.lat(), dest_pt.lng())
                            .Union()
                      : CreateInputStation(fbb, fbb.CreateString(to->eva_nr_),
                                           fbb.CreateString(""))
                            .Union(),
                  fbb.CreateVector(create_modes(fbb, dest_modes)),
                  SearchType_Default, dir, fbb.CreateString(router))
                  .Union(),
              "/intermodal", DestinationType_Module, id);
        }

        break;
      }

      case IntermodalStart_OntripTrainStart: {
        auto const [trip, trip_stop] = random_trip_and_station(sched);

        auto const& primary = trip->id_.primary_;
        auto const& secondary = trip->id_.secondary_;

        auto const& primary_station_eva =
            sched.stations_[primary.get_station_id()]->eva_nr_;
        auto const& target_station_eva =
            sched.stations_[secondary.target_station_id_]->eva_nr_;

        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_IntermodalRoutingRequest,
              CreateIntermodalRoutingRequest(
                  fbb, start_type,
                  CreateOntripTrainStart(
                      fbb,
                      CreateTripId(
                          fbb, fbb.CreateString(trip->gtfs_trip_id_),
                          fbb.CreateString(primary_station_eva),
                          primary.get_train_nr(),
                          motis_to_unixtime(sched, primary.get_time()),
                          fbb.CreateString(target_station_eva),
                          motis_to_unixtime(sched, secondary.target_time_),
                          fbb.CreateString(secondary.line_id_)),
                      CreateInputStation(
                          fbb,
                          fbb.CreateString(
                              trip_stop.get_station(sched).eva_nr_),
                          fbb.CreateString("")),
                      motis_to_unixtime(sched, trip_stop.arr_lcon().a_time_))
                      .Union(),
                  fbb.CreateVector(create_modes(fbb, start_modes)),
                  destination_type, get_destination(fbb),
                  fbb.CreateVector(create_modes(fbb, dest_modes)),
                  SearchType_Default, dir, fbb.CreateString(router))
                  .Union(),
              "/intermodal", DestinationType_Module, id);
        }

        break;
      }

      case IntermodalStart_OntripStationStart:
        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_IntermodalRoutingRequest,
              CreateIntermodalRoutingRequest(
                  fbb, start_type,
                  CreateOntripStationStart(
                      fbb,
                      CreateInputStation(fbb, fbb.CreateString(from->eva_nr_),
                                         fbb.CreateString("")),
                      interval.begin())
                      .Union(),
                  fbb.CreateVector(create_modes(fbb, start_modes)),
                  destination_type, get_destination(fbb),
                  fbb.CreateVector(create_modes(fbb, dest_modes)),
                  SearchType_Default, dir, fbb.CreateString(router))
                  .Union(),
              "/intermodal", DestinationType_Module, id);
        }
        break;

      case IntermodalStart_PretripStart:
        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_IntermodalRoutingRequest,
              CreateIntermodalRoutingRequest(
                  fbb, start_type,
                  CreatePretripStart(
                      fbb,
                      CreateInputStation(fbb, fbb.CreateString(from->eva_nr_),
                                         fbb.CreateString("")),
                      &interval, min_connection_count, extend_earlier,
                      extend_later)
                      .Union(),
                  fbb.CreateVector(create_modes(fbb, start_modes)),
                  destination_type, get_destination(fbb),
                  fbb.CreateVector(create_modes(fbb, dest_modes)),
                  SearchType_Default, dir, fbb.CreateString(router))
                  .Union(),
              "/intermodal", DestinationType_Module, id);
        }
        break;

      default:
        throw utl::fail(
            "start type {} not supported for message type intermodal",
            EnumNameIntermodalStart(start_type));
    }
  } else if (message_type == MsgContent_RoutingRequest) {
    using namespace motis::routing;

    switch (start_type) {
      case IntermodalStart_OntripTrainStart: {
        auto const [trip, trip_stop] = random_trip_and_station(sched);

        auto const& primary = trip->id_.primary_;
        auto const& secondary = trip->id_.secondary_;

        auto const& primary_station_eva =
            sched.stations_[primary.get_station_id()]->eva_nr_;
        auto const& target_station_eva =
            sched.stations_[secondary.target_station_id_]->eva_nr_;

        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_RoutingRequest,
              motis::routing::CreateRoutingRequest(
                  fbb, Start_OntripTrainStart,
                  CreateOntripTrainStart(
                      fbb,
                      CreateTripId(
                          fbb, fbb.CreateString(trip->gtfs_trip_id_),
                          fbb.CreateString(primary_station_eva),
                          primary.get_train_nr(),
                          motis_to_unixtime(sched, primary.get_time()),
                          fbb.CreateString(target_station_eva),
                          motis_to_unixtime(sched, secondary.target_time_),
                          fbb.CreateString(secondary.line_id_)),
                      CreateInputStation(
                          fbb,
                          fbb.CreateString(
                              trip_stop.get_station(sched).eva_nr_),
                          fbb.CreateString("")),
                      motis_to_unixtime(sched, trip_stop.arr_lcon().a_time_))
                      .Union(),
                  CreateInputStation(fbb, fbb.CreateString(to->eva_nr_),
                                     fbb.CreateString("")),
                  SearchType_Default, dir,
                  fbb.CreateVector(std::vector<fbs::Offset<Via>>()),
                  fbb.CreateVector(
                      std::vector<fbs::Offset<AdditionalEdgeWrapper>>()))
                  .Union(),
              router, DestinationType_Module, id);
        }

        break;
      }

      case IntermodalStart_OntripStationStart:
        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_RoutingRequest,
              CreateRoutingRequest(
                  fbb, Start_OntripStationStart,
                  CreateOntripStationStart(
                      fbb,
                      CreateInputStation(fbb, fbb.CreateString(from->eva_nr_),
                                         fbb.CreateString("")),
                      interval.begin())
                      .Union(),
                  CreateInputStation(fbb, fbb.CreateString(to->eva_nr_),
                                     fbb.CreateString("")),
                  SearchType_Default, dir,
                  fbb.CreateVector(std::vector<fbs::Offset<Via>>()),
                  fbb.CreateVector(
                      std::vector<fbs::Offset<AdditionalEdgeWrapper>>()))
                  .Union(),
              router, DestinationType_Module, id);
        }
        break;

      case IntermodalStart_PretripStart:
        for (auto const& [fbbp, router] : utl::zip(fbbs, routers)) {
          auto& fbb = *fbbp;
          fbb.create_and_finish(
              MsgContent_RoutingRequest,
              CreateRoutingRequest(
                  fbb, Start_PretripStart,
                  CreatePretripStart(
                      fbb,
                      CreateInputStation(fbb, fbb.CreateString(from->eva_nr_),
                                         fbb.CreateString("")),
                      &interval, min_connection_count, extend_earlier,
                      extend_later)
                      .Union(),
                  CreateInputStation(fbb, fbb.CreateString(to->eva_nr_),
                                     fbb.CreateString("")),
                  SearchType_Default, dir,
                  fbb.CreateVector(std::vector<fbs::Offset<Via>>()),
                  fbb.CreateVector(
                      std::vector<fbs::Offset<AdditionalEdgeWrapper>>()))
                  .Union(),
              router, DestinationType_Module, id);
        }
        break;

      default:
        throw utl::fail("start type {} not supported for message type routing",
                        EnumNameIntermodalStart(start_type));
    }
  }

  for (auto const& [out_file, fbbp] : utl::zip(out_files, fbbs)) {
    auto& fbb = *fbbp;
    out_file << make_msg(fbb)->to_json(json_format::SINGLE_LINE) << "\n";
  }
}

bool has_events(edge const& e, motis::time const from, motis::time const to) {
  auto con = e.get_connection(from);
  return con != nullptr && con->d_time_ <= to;
}

bool has_events(station_node const& s, motis::time const from,
                motis::time const to) {
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
                      unixtime const motis_interval_start,
                      unixtime const motis_interval_end) {
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
    Interval const interval) {
  station const *from = nullptr, *to = nullptr;
  auto motis_interval_start = unix_to_motistime(sched, interval.begin());
  auto motis_interval_end = unix_to_motistime(sched, interval.end());
  if (motis_interval_start == INVALID_TIME ||
      motis_interval_end == INVALID_TIME) {
    std::cout << "ERROR: generated timestamp not valid:\n";
    std::cout << "  schedule range: " << sched.schedule_begin_ << " - "
              << sched.schedule_end_ << "\n";
    std::cout << "  interval_start = " << interval.begin() << " ("
              << format_time(motis_interval_start) << ")\n";
    std::cout << "  interval_end = " << interval.end() << " ("
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

int generate(int argc, char const** argv) {
  using namespace motis::intermodal;
  using namespace motis::intermodal::eval;
  dataset_settings dataset_opt;
  generator_settings generator_opt;
  import_settings import_opt;

  try {
    conf::options_parser parser({&dataset_opt, &generator_opt, &import_opt});
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

  auto const start_modes = read_modes(generator_opt.start_modes_);
  auto const dest_modes = read_modes(generator_opt.dest_modes_);
  auto const start_type = generator_opt.get_start_type();
  auto const dest_type = generator_opt.get_dest_type();
  auto const message_type = generator_opt.get_message_type();

  utl::verify(generator_opt.dest_type_ == "coordinate" ||
                  generator_opt.dest_type_ == "station",
              "unknown destination type {}, supported: coordinate, station",
              generator_opt.dest_type_);
  utl::verify(!start_modes.empty() ||
                  (start_type != IntermodalStart_IntermodalOntripStart &&
                   start_type != IntermodalStart_IntermodalPretripStart),
              "no start modes given: {} (start_type={})",
              generator_opt.start_modes_, EnumNameIntermodalStart(start_type));
  utl::verify(
      !dest_modes.empty() || dest_type != IntermodalDestination_InputPosition,
      "no destination modes given: {}, dest_type={}", generator_opt.dest_modes_,
      EnumNameIntermodalDestination(dest_type));

  auto bds = parse_bounds(generator_opt);
  auto of_streams =
      utl::to_vec(generator_opt.routers_, [&](std::string const& router) {
        return std::ofstream{replace_target_escape(generator_opt.out_, router)};
      });

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt, import_opt);

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

  auto const start_radius = get_radius_in_m(start_modes);
  auto const dest_radius = get_radius_in_m(dest_modes);

  for (int i = 1; i <= generator_opt.query_count_; ++i) {
    if ((i % 100) == 0) {
      std::cout << i << "/" << generator_opt.query_count_ << std::endl;
    }

    auto const interval = interval_gen.random_interval();
    auto const [from, to] = random_stations(sched, station_nodes, interval);

    write_query(sched, point_gen, i, interval, from, to, start_modes,
                dest_modes, start_radius, dest_radius, message_type, start_type,
                dest_type, generator_opt.get_search_dir(),
                generator_opt.extend_earlier_, generator_opt.extend_later_,
                generator_opt.min_connection_count_, generator_opt.routers_,
                of_streams);
  }

  return 0;
}

}  // namespace motis::intermodal::eval
