#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/module/message.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/lift.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"
#include "gtest/gtest.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt;
using motis::test::schedule::simple_realtime::eval_opt;

namespace motis {
namespace ridesharing {

struct eval_super_itest : public motis_instance_test {
  eval_super_itest()
      : motis::test::motis_instance_test(
            {"../../rohdaten", "20191013", 16, false, true, false, true},
            {"lookup", "ridesharing", "osrm", "ppr", "parking", "intermodal",
             "routing"},
            {"--ridesharing.database_path=rs.mdb",
             "--osrm.dataset=car/germany-latest.osrm",
             "--ridesharing.use_parking=true", "--ppr.graph=germany.ppr",
             "--ppr.profile=deps/ppr/profiles/default.json",
             "--parking.parking=parking.txt", "--parking.db_max_size=1048576",
             "--parking.db=parking_footedges.db"}) {}

  msg_ptr ridesharing_remove(int driver, int time_lift_start) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_RidesharingRemove,
        CreateRidesharingRemove(mc, driver, time_lift_start).Union(),
        "/ridesharing/remove");
    return make_msg(mc);
  }

  msg_ptr ridesharing_create(int driver, long time_lift_start,
                             geo::latlng const& start,
                             geo::latlng const& dest) {
    message_creator mc;
    Position s{start.lat_, start.lng_};
    Position d{dest.lat_, dest.lng_};

    mc.create_and_finish(
        MsgContent_RidesharingCreate,
        CreateRidesharingCreate(mc, driver, time_lift_start, 4, &s, &d).Union(),
        "/ridesharing/create");
    return make_msg(mc);
  }

  msg_ptr ridesharing_edges(long t, geo::latlng const& s,
                            geo::latlng const& d) {
    message_creator mc;
    Position start{s.lat_, s.lng_};
    Position destination{d.lat_, d.lng_};
    std::string profile{"default"};

    mc.create_and_finish(
        MsgContent_RidesharingRequest,
        CreateRidesharingRequest(mc, &start, &destination, t, 1, QUERYMODE_BOTH,
                                 motis::ppr::CreateSearchOptions(
                                     mc, mc.CreateString(profile), 1000000))
            .Union(),
        "/ridesharing/edges");
    return make_msg(mc);
  }

  msg_ptr ridesharing_stats() { return make_no_msg("/ridesharing/stats"); }

  msg_ptr ridesharing_book(int driver, int time_lift_start, int passenger,
                           geo::latlng const& piu, geo::latlng const& dro,
                           uint16_t from_leg, uint16_t to_leg,
                           long const required_arrival = 0) {
    message_creator mc;
    Position pick_up{piu.lat_, piu.lng_};
    Position drop_off{dro.lat_, dro.lng_};

    mc.create_and_finish(
        MsgContent_RidesharingBook,
        CreateRidesharingBook(mc, driver, time_lift_start, passenger, 1,
                              required_arrival, &pick_up, from_leg, &drop_off,
                              to_leg, 250)
            .Union(),
        "/ridesharing/book");
    return make_msg(mc);
  }

  msg_ptr execute_intermodal_pure(time_t t, geo::latlng const& qs,
                                  geo::latlng const& qd) {
    message_creator mc;
    Position query_start{qs.lat_, qs.lng_};
    auto const dest_loc = geo::latlng{qd.lat_, qd.lng_};
    std::vector<Offset<ModeWrapper>> start_modes{CreateModeWrapper(
        mc, Mode_FootPPR,
        CreateFootPPR(mc, motis::ppr::CreateSearchOptions(
                              mc, mc.CreateString("default"), 30 * 60))
            .Union())};
    std::vector<Offset<ModeWrapper>> end_modes{CreateModeWrapper(
        mc, Mode_FootPPR,
        CreateFootPPR(mc, motis::ppr::CreateSearchOptions(
                              mc, mc.CreateString("default"), 30 * 60))
            .Union())};
    mc.create_and_finish(
        MsgContent_IntermodalRoutingRequest,
        CreateIntermodalRoutingRequest(
            mc, IntermodalStart_IntermodalOntripStart,
            CreateIntermodalOntripStart(mc, &query_start, t).Union(),
            mc.CreateVector(start_modes), IntermodalDestination_InputPosition,
            CreateInputPosition(mc, dest_loc.lat_, dest_loc.lng_).Union(),
            mc.CreateVector(end_modes), SearchType_DefaultPrice,
            SearchDir_Forward)
            .Union(),
        "/intermodal");
    return make_msg(mc);
  }

  msg_ptr execute_intermodal(time_t t, geo::latlng const& qs,
                             geo::latlng const& qd) {
    message_creator mc;
    Position query_start{qs.lat_, qs.lng_};
    auto const dest_loc = geo::latlng{qd.lat_, qd.lng_};
    // LOG(logging::info) << "From: " << qs.lat_ << "/" << qs.lng_ << " -> To: "
    // << qd.lat_ << "/" << qd.lng_;
    std::vector<Offset<ModeWrapper>> start_modes{
        CreateModeWrapper(
            mc, Mode_Ridesharing,
            CreateRidesharing(mc, 1,
                              motis::ppr::CreateSearchOptions(
                                  mc, mc.CreateString("default"), 30 * 60))
                .Union()),
        CreateModeWrapper(
            mc, Mode_FootPPR,
            CreateFootPPR(mc, motis::ppr::CreateSearchOptions(
                                  mc, mc.CreateString("default"), 30 * 60))
                .Union())};
    std::vector<Offset<ModeWrapper>> end_modes{
        CreateModeWrapper(
            mc, Mode_Ridesharing,
            CreateRidesharing(mc, 1,
                              motis::ppr::CreateSearchOptions(
                                  mc, mc.CreateString("default"), 30 * 60))
                .Union()),
        CreateModeWrapper(
            mc, Mode_FootPPR,
            CreateFootPPR(mc, motis::ppr::CreateSearchOptions(
                                  mc, mc.CreateString("default"), 30 * 60))
                .Union())};
    mc.create_and_finish(
        MsgContent_IntermodalRoutingRequest,
        CreateIntermodalRoutingRequest(
            mc, IntermodalStart_IntermodalOntripStart,
            CreateIntermodalOntripStart(mc, &query_start, t).Union(),
            mc.CreateVector(start_modes), IntermodalDestination_InputPosition,
            CreateInputPosition(mc, dest_loc.lat_, dest_loc.lng_).Union(),
            mc.CreateVector(end_modes), SearchType_DefaultPrice,
            SearchDir_Forward)
            .Union(),
        "/intermodal");
    return make_msg(mc);
  }
};

}  // namespace ridesharing
}  // namespace motis
