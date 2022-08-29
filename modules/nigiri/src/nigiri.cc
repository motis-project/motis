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

#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/module/event_collector.h"

#include "motis/nigiri/nigiri_to_motis_journey.h"
#include "motis/nigiri/unixtime_conv.h"

namespace fs = std::filesystem;
namespace mm = motis::module;
namespace n = ::nigiri;
namespace fbs = flatbuffers;

namespace motis::nigiri {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

struct nigiri::impl {
  std::shared_ptr<n::timetable> tt_;
  std::vector<std::string> tags_;
};

n::location_id motis_station_to_nigiri_id(std::vector<std::string> const& tags,
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
  return tt.locations_.location_id_to_idx_.at(
      motis_station_to_nigiri_id(tags, station_id));
}

nigiri::nigiri() : module("Next Generation Routing", "nigiri") {}

nigiri::~nigiri() = default;

mm::msg_ptr to_routing_response(
    n::timetable const& tt, std::vector<std::string> const& tags,
    n::pareto_set<n::routing::journey> const& journeys,
    n::interval<n::unixtime_t> search_interval) {
  mm::message_creator fbb;
  std::vector<flatbuffers::Offset<Statistics>> stats{};
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      routing::CreateRoutingResponse(
          fbb, fbb.CreateVectorOfSortedTables(&stats),
          fbb.CreateVector(utl::to_vec(
              journeys,
              [&](n::routing::journey const& j) {
                return to_connection(fbb, nigiri_to_motis_journey(tt, tags, j));
              })),
          to_motis_unixtime(search_interval.from_),
          to_motis_unixtime(search_interval.to_))
          .Union());
  return make_msg(fbb);
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

        if (search_state.get() == nullptr) {
          search_state.reset(new n::routing::search_state{});
        }

        auto tt = impl_->tt_;
        if (req->search_dir() == SearchDir_Forward) {
          n::routing::raptor<n::direction::kForward>{tt, *search_state,
                                                     std::move(q)}
              .route();
        } else {
          n::routing::raptor<n::direction::kBackward>{tt, *search_state,
                                                      std::move(q)}
              .route();
        }

        return to_routing_response(*tt, impl_->tags_, search_state->results_,
                                   search_state->search_interval_);
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
