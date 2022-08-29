#include "motis/nigiri/nigiri.h"

#include "boost/thread/tss.hpp"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor.h"
#include "nigiri/routing/search_state.h"

#include "motis/core/common/interval_map.h"
#include "motis/core/journey/journey.h"
#include "motis/module/event_collector.h"
#include "utl/pairwise.h"
#include "utl/parser/csv.h"

namespace fs = std::filesystem;
namespace fbs = flatbuffers;
namespace mm = motis::module;
namespace n = ::nigiri;

namespace motis::nigiri {

boost::thread_specific_ptr<n::routing::search_state> search_state_tsp;

struct nigiri::impl {
  std::shared_ptr<n::timetable> tt_;
  std::vector<std::string> tags_;
};

nigiri::nigiri() : module("Next Generation Routing", "nigiri") {}

nigiri::~nigiri() = default;

n::location_id get_id(std::vector<std::string> const& tags,
                      std::string const& station_id) {
  auto const it = utl::find_if(
      tags, [&](auto&& tag) { return station_id.starts_with(tag); });
  return it == end(tags)
             ? n::location_id{station_id, n::source_idx_t{0U}}
             : n::location_id{
                   station_id.substr(it->length()),
                   n::source_idx_t{static_cast<cista::base_t<n::source_idx_t>>(
                       std::distance(begin(tags), it))}};
}

n::location_idx_t get_location_idx(std::vector<std::string> const& tags,
                                   n::timetable const& tt,
                                   std::string const& station_id) {
  return tt.locations_.location_id_to_idx_.at(get_id(tags, station_id));
}

unixtime to_motis_unixtime(n::unixtime_t const t) {
  return motis::unixtime{
      std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch())
          .count()};
}

mcd::string get_station_id(std::vector<std::string> const& tags,
                           n::timetable const& tt, n::location_idx_t const l) {
  return tags.at(to_idx(tt.locations_.src_.at(l))) +
         tt.locations_.ids_.at(l).str();
}

extern_trip nigiri_trip_to_extern_trip(std::vector<std::string> const& tags,
                                       n::timetable const& tt,
                                       n::trip_idx_t const trip,
                                       n::day_idx_t const day) {
  auto const [transport, stop_range] = tt.trip_ref_transport_[trip];
  auto const first_location =
      tt.route_location_seq_[tt.transport_route_[transport]]
          .front()
          .location_idx();
  auto const last_location =
      tt.route_location_seq_[tt.transport_route_[transport]]
          .back()
          .location_idx();
  auto const id = tt.trip_ids_.at(trip).back();
  auto const [admin, train_nr, first_stop_eva, fist_start_time, last_stop_eva,
              last_stop_time, line] =
      utl::split<'/', utl::cstr, unsigned, utl::cstr, unsigned, utl::cstr,
                 unsigned, utl::cstr>(utl::cstr{id.id_});
  return extern_trip{
      .station_id_ = get_station_id(tags, tt, first_location),
      .train_nr_ = train_nr,
      .time_ = to_motis_unixtime(tt.event_time(
          {transport, day}, stop_range.from_, n::event_type::kDep)),
      .target_station_id_ = get_station_id(tags, tt, last_location),
      .target_time_ = to_motis_unixtime(
          tt.event_time({transport, day}, stop_range.to_, n::event_type::kArr)),
      .line_id_ = line.to_str()};
}

journey nigiri_to_motis_journey(n::timetable const& tt,
                                std::vector<std::string> const& tags,
                                n::routing::journey const& nj) {
  journey mj;

  auto const fill_stop_info = [&](motis::journey::stop& s,
                                  n::location_idx_t const l) {
    auto const& l_name = tt.locations_.names_.at(l);
    auto const& pos = tt.locations_.coordinates_.at(l);
    s.name_ = l_name.str();
    s.eva_no_ = get_station_id(tags, tt, l);
    s.lat_ = pos.lat_;
    s.lng_ = pos.lng_;
  };

  auto const add_walk = [&](n::routing::journey::leg const& leg, int mumo_id) {
    auto& from_stop =
        mj.stops_.empty() ? mj.stops_.emplace_back() : mj.stops_.back();
    auto const from_idx = static_cast<unsigned>(mj.stops_.size() - 1);
    fill_stop_info(from_stop, leg.from_);

    auto& to_stop = mj.stops_.emplace_back();
    auto const to_idx = static_cast<unsigned>(mj.stops_.size() - 1);
    fill_stop_info(to_stop, leg.to_);

    auto t = journey::transport{};
    t.from_ = from_idx;
    t.to_ = to_idx;
    t.is_walk_ = true;
    t.duration_ =
        static_cast<unsigned>((leg.arr_time_ - leg.dep_time_).count());
    t.mumo_id_ = mumo_id;
    mj.transports_.emplace_back(std::move(t));
  };

  for (auto const& leg : nj.legs_) {
    leg.uses_.apply(utl::overloaded{
        [&](n::routing::journey::transport_enter_exit const& t) {
          auto const& route_idx = tt.transport_route_.at(t.t_.t_idx_);
          auto const& stop_seq = tt.route_location_seq_.at(route_idx);

          interval_map<journey::transport> transports;
          interval_map<extern_trip> trips;

          //          (void)transports;
          //          (void)trips;
          //          for (auto const& section : utl::pairwise(t.stop_range_)) {
          //            (void)section;
          //          }

          for (auto const& stop_idx : t.stop_range_) {
            auto const exit = (stop_idx == t.stop_range_.to_ - 1U);
            auto const enter = (stop_idx == t.stop_range_.from_);

            // for entering: create a new stop if it's the first stop in journey
            // otherwise: create a new stop
            auto const reuse_arrival = enter && !mj.stops_.empty();
            auto& stop =
                reuse_arrival ? mj.stops_.back() : mj.stops_.emplace_back();
            fill_stop_info(stop, stop_seq.at(stop_idx).location_idx());

            if (exit) {
              stop.exit_ = true;
            }
            if (enter) {
              stop.enter_ = true;
            }

            if (!enter) {
              auto const time = to_motis_unixtime(
                  tt.event_time(t.t_, stop_idx, n::event_type::kArr));
              stop.arrival_ = journey::stop::event_info{
                  .valid_ = true,
                  .timestamp_ = time,
                  .schedule_timestamp_ = time,
                  .timestamp_reason_ = timestamp_reason::SCHEDULE,
                  .track_ = "",
                  .schedule_track_ = ""};
            }

            if (!exit) {
              auto const time = to_motis_unixtime(
                  tt.event_time(t.t_, stop_idx, n::event_type::kDep));
              stop.departure_ = journey::stop::event_info{
                  .valid_ = true,
                  .timestamp_ = time,
                  .schedule_timestamp_ = time,
                  .timestamp_reason_ = timestamp_reason::SCHEDULE,
                  .track_ = "",
                  .schedule_track_ = ""};
            }
          }
        },
        [&](n::footpath_idx_t const) { add_walk(leg, -1); },
        [&](std::uint8_t const x) { add_walk(leg, x); }});
  }
  return mj;
}

mm::msg_ptr to_routing_response(
    n::timetable const& tt,
    n::pareto_set<n::routing::journey> const& journeys) {
  (void)tt;
  (void)journeys;
  mm::message_creator mc;
  //  routing::CreateRoutingResponse();
  return make_msg(mc);
}

void nigiri::init(motis::module::registry& reg) {
  reg.register_op(
      "/nigiri/routing",
      [&](mm::msg_ptr const& msg) {
        using motis::routing::RoutingRequest;
        auto const req = motis_content(RoutingRequest, msg);

        utl::verify(req->start_type() == routing::Start_PretripStart,
                    "nigiri currently only supports pre-trip queries");

        auto const start =
            reinterpret_cast<routing::PretripStart const*>(req->start());
        utl::verify(start->min_connection_count() == 0U &&
                        !start->extend_interval_earlier() &&
                        !start->extend_interval_later(),
                    "nigiri currently does not support interval extension");

        auto q = n::routing::query{
            .interval_ =
                {n::unixtime_t{std::chrono::duration_cast<n::i32_minutes>(
                     std::chrono::seconds{start->interval()->begin()})},
                 n::unixtime_t{std::chrono::duration_cast<n::i32_minutes>(
                     std::chrono::seconds{start->interval()->end()})}},
            .start_ = {n::routing::offset{
                .location_ = get_location_idx(impl_->tags_, *impl_->tt_,
                                              start->station()->id()->str()),
                .offset_ = n::duration_t{0U},
                .type_ = 0U}},
            .destinations_ = {n::vector<n::routing::offset>{n::routing::offset{
                .location_ = get_location_idx(impl_->tags_, *impl_->tt_,
                                              req->destination()->id()->str()),
                .offset_ = n::duration_t{0U},
                .type_ = 0U}}},
            .via_destinations_ = {},
            .allowed_classes_ = cista::bitset<n::kNumClasses>{},
            .max_transfers_ = n::routing::kMaxTransfers,
            .min_connection_count_ = 0U,
            .extend_interval_earlier_ = false,
            .extend_interval_later_ = false};

        if (search_state_tsp.get() == nullptr) {
          search_state_tsp.reset(new n::routing::search_state{});
        }

        auto tt = impl_->tt_;
        switch (req->search_dir()) {
          case SearchDir_Forward:
            n::routing::raptor<n::direction::kForward>{
                tt, *search_state_tsp.get(), std::move(q)}
                .route();
            break;

          case SearchDir_Backward:
            n::routing::raptor<n::direction::kBackward>{
                tt, *search_state_tsp.get(), std::move(q)}
                .route();
            break;
        }

        return to_routing_response(*tt, search_state_tsp.get()->results_);
      },
      {});
}

void nigiri::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "nigiri", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        auto const& msg = dependencies.at("SCHEDULE");

        impl_ = std::make_unique<impl>();
        impl_->tt_ = std::make_shared<n::timetable>();

        using import::FileEvent;
        for (auto const [i, p] :
             utl::enumerate(*motis_content(FileEvent, msg)->paths())) {
          if (p->tag()->str() != "schedule") {
            continue;
          }
          auto const path = fs::path{p->path()->str()};
          auto const d = n::loader::make_dir(path);
          auto const c = utl::find_if(n::loader::hrd::configs, [&](auto&& c) {
            return n::loader::hrd::applicable(c, *d);
          });
          utl::verify(c != end(n::loader::hrd::configs),
                      "no loader applicable to {}", path);
          n::loader::hrd::load_timetable(n::source_idx_t{i}, *c, *d,
                                         *impl_->tt_);
          impl_->tags_.emplace_back(p->tag()->str() + "-");
        }
      })
      ->require("SCHEDULE", [](mm::msg_ptr const& msg) {
        if (msg->get()->content_type() != MsgContent_FileEvent) {
          return false;
        }
        using import::FileEvent;
        return utl::all_of(
            *motis_content(FileEvent, msg)->paths(),
            [](import::ImportPath const* p) {
              if (p->tag()->str() != "schedule") {
                return true;
              }
              auto const d = n::loader::make_dir(fs::path{p->path()->str()});
              return utl::any_of(n::loader::hrd::configs, [&](auto&& c) {
                return n::loader::hrd::applicable(c, *d);
              });
            });
      });
}

}  // namespace motis::nigiri
