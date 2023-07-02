#include "motis/lookup/lookup.h"

#include "utl/enumerate.h"

#include "motis/core/access/time_access.h"

#include "motis/module/event_collector.h"

#include "motis/core/access/station_access.h"
#include "motis/lookup/error.h"
#include "motis/lookup/lookup_geo_station.h"
#include "motis/lookup/lookup_id_train.h"
#include "motis/lookup/lookup_meta_station.h"
#include "motis/lookup/lookup_ribasis.h"
#include "motis/lookup/lookup_station_events.h"
#include "motis/lookup/lookup_station_info.h"

using namespace flatbuffers;
using namespace motis::module;

namespace motis::lookup {

lookup::lookup() : module("Lookup", "lookup") {}
lookup::~lookup() = default;

void lookup::init(registry& r) {
  auto const& sched = get_sched();
  station_geo_index_ = std::make_unique<geo::point_rtree>(
      geo::make_point_rtree(sched.stations_, [](auto const& s) {
        return geo::latlng{s->lat(), s->lng()};
      }));

  r.register_op("/lookup/geo_station_id",
                [&](msg_ptr const& m) { return lookup_station_id(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/geo_station",
                [&](msg_ptr const& m) { return lookup_station(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/geo_station_batch",
                [&](msg_ptr const& m) { return lookup_stations(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/station_events",
                [&](msg_ptr const& m) { return lookup_station_events(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/schedule_info",
                [&](msg_ptr const&) { return lookup_schedule_info(); },
                {kScheduleReadAccess});
  r.register_op("/lookup/id_train",
                [&](msg_ptr const& m) { return lookup_id_train(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/meta_station",
                [&](msg_ptr const& m) { return lookup_meta_station(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/meta_station_batch",
                [&](msg_ptr const& m) { return lookup_meta_stations(m); },
                {kScheduleReadAccess});
  r.register_op("/lookup/ribasis",
                [&](msg_ptr const& m) { return lookup_ribasis(m); }, {});
  r.register_op("/lookup/station_info",
                [&](msg_ptr const& m) { return lookup_station_info(m); }, {});
  r.register_op("/lookup/station_location",
                [&](msg_ptr const& m) { return lookup_station_location(m); },
                {kScheduleReadAccess});
}

void lookup::import(motis::module::import_dispatcher& reg) {
  std::make_shared<motis::module::event_collector>(
      get_data_directory().generic_string(), "lookup", reg,
      [this](motis::module::event_collector::dependencies_map_t const&,
             motis::module::event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE", [](motis::module::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

bool lookup::import_successful() const { return import_successful_; }

msg_ptr lookup::lookup_station_id(msg_ptr const& msg) const {
  auto req = motis_content(LookupGeoStationIdRequest, msg);

  message_creator b;
  auto response = motis::lookup::lookup_geo_stations_id(b, *station_geo_index_,
                                                        get_sched(), req);
  b.create_and_finish(MsgContent_LookupGeoStationResponse, response.Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_station(msg_ptr const& msg) const {
  auto req = motis_content(LookupGeoStationRequest, msg);

  message_creator b;
  auto response = motis::lookup::lookup_geo_stations(b, *station_geo_index_,
                                                     get_sched(), req);
  b.create_and_finish(MsgContent_LookupGeoStationResponse, response.Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_stations(msg_ptr const& msg) const {
  auto req = motis_content(LookupBatchGeoStationRequest, msg);

  message_creator b;
  std::vector<Offset<LookupGeoStationResponse>> responses;
  for (auto const& sub_req : *req->requests()) {
    responses.push_back(motis::lookup::lookup_geo_stations(
        b, *station_geo_index_, get_sched(), sub_req));
  }
  b.create_and_finish(
      MsgContent_LookupBatchGeoStationResponse,
      CreateLookupBatchGeoStationResponse(b, b.CreateVector(responses))
          .Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_station_events(msg_ptr const& msg) {
  auto req = motis_content(LookupStationEventsRequest, msg);

  message_creator b;
  auto const& sched = get_sched();
  auto events = motis::lookup::lookup_station_events(b, sched, req);
  b.create_and_finish(
      MsgContent_LookupStationEventsResponse,
      CreateLookupStationEventsResponse(b, b.CreateVector(events)).Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_id_train(msg_ptr const& msg) {
  auto req = motis_content(LookupIdTrainRequest, msg);

  message_creator b;
  auto const& sched = get_sched();
  auto train = motis::lookup::lookup_id_train(b, sched, req->trip_id());
  b.create_and_finish(MsgContent_LookupIdTrainResponse,
                      CreateLookupIdTrainResponse(b, train).Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_meta_station(msg_ptr const& msg) {
  auto req = motis_content(LookupMetaStationRequest, msg);

  message_creator b;
  auto const& sched = get_sched();
  b.create_and_finish(
      MsgContent_LookupMetaStationResponse,
      motis::lookup::lookup_meta_station(b, sched, req).Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_meta_stations(msg_ptr const& msg) {
  auto req = motis_content(LookupBatchMetaStationRequest, msg);

  message_creator b;
  auto const& sched = get_sched();
  std::vector<Offset<LookupMetaStationResponse>> responses;
  for (auto const& r : *req->requests()) {
    responses.push_back(motis::lookup::lookup_meta_station(b, sched, r));
  }
  b.create_and_finish(
      MsgContent_LookupBatchMetaStationResponse,
      CreateLookupBatchMetaStationResponse(b, b.CreateVector(responses))
          .Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_schedule_info() {
  auto const& sched = get_sched();

  std::stringstream ss;
  for (auto const& [i, name] : utl::enumerate(sched.names_)) {
    if (i != 0) {
      ss << "\n";
    }
    ss << name;
  }

  message_creator b;
  b.create_and_finish(
      MsgContent_LookupScheduleInfoResponse,
      CreateLookupScheduleInfoResponse(b, b.CreateString(ss.str()),
                                       external_schedule_begin(sched),
                                       external_schedule_end(sched))
          .Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_station_location(msg_ptr const& msg) {
  using namespace motis::routing;
  auto const req = motis_content(InputStation, msg);

  auto const& sched = get_sched();
  auto const station = get_station(sched, req->id()->str());
  auto const pos = Position{station->lat(), station->lng()};

  message_creator b;
  b.create_and_finish(MsgContent_LookupStationLocationResponse,
                      CreateLookupStationLocationResponse(b, &pos).Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_station_info(msg_ptr const& msg) {
  auto req = motis_content(LookupStationInfoRequest, msg);
  auto const schedule_res_id =
      req->schedule() == 0U ? to_res_id(global_res_id::SCHEDULE)
                            : static_cast<ctx::res_id_t>(req->schedule());
  auto res_lock = lock_resources({{schedule_res_id, ctx::access_t::READ}});
  auto const& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;

  message_creator b;
  auto const res = motis::lookup::lookup_station_info(b, sched, req);
  b.create_and_finish(MsgContent_LookupStationInfoResponse, res.Union());
  return make_msg(b);
}

msg_ptr lookup::lookup_ribasis(msg_ptr const& msg) {
  auto req = motis_content(LookupRiBasisRequest, msg);
  auto const schedule_res_id =
      req->schedule() == 0U ? to_res_id(global_res_id::SCHEDULE)
                            : static_cast<ctx::res_id_t>(req->schedule());
  auto res_lock = lock_resources({{schedule_res_id, ctx::access_t::READ}});
  auto const& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;

  message_creator b;
  auto rbf = motis::lookup::lookup_ribasis(b, sched, req);
  b.create_and_finish(MsgContent_LookupRiBasisResponse, rbf.Union());
  return make_msg(b);
}

}  // namespace motis::lookup
