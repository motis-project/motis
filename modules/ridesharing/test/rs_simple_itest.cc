#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"

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
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::ridesharing {

struct rs_simple_itest : public motis_instance_test {
  rs_simple_itest()
      : motis::test::motis_instance_test(
            dataset_opt, {"lookup", "ridesharing"},
            {"--ridesharing.database_path=:memory:",
             "--ridesharing.use_parking=false"}) {
    instance_->register_op("/osrm/many_to_many", [&](msg_ptr const& msg) {
      auto const req = motis_content(OSRMManyToManyRequest, msg);
      std::vector<latlng> coor;
      for (auto const& loc : *req->coordinates()) {
        coor.emplace_back(latlng{loc->lat(), loc->lng()});
      }

      message_creator mc;
      std::vector<Offset<OSRMManyToManyResponseRow>> routing_matrix;
      routing_matrix.reserve(coor.size());
      for ([[maybe_unused]] auto const& e : coor) {
        std::vector<Cost> cost_row;
        cost_row.reserve(coor.size());
        for ([[maybe_unused]] auto const& f : coor) {
          cost_row.emplace_back(100000, 100000);
        }
        routing_matrix.emplace_back(CreateOSRMManyToManyResponseRow(
            mc, mc.CreateVectorOfStructs(cost_row)));
      }

      mc.create_and_finish(
          MsgContent_OSRMManyToManyResponse,
          CreateOSRMManyToManyResponse(mc, mc.CreateVector(routing_matrix))
              .Union());
      return make_msg(mc);
    });

    instance_->register_op("/osrm/one_to_many", [&](msg_ptr const& msg) {
      auto const req = motis_content(OSRMOneToManyRequest, msg);

      std::vector<Cost> costs;
      costs.reserve(req->many()->size());
      for ([[maybe_unused]] auto const& e : *req->many()) {
        costs.emplace_back(100000, 100000);
      }

      message_creator mc;
      mc.create_and_finish(
          MsgContent_OSRMOneToManyResponse,
          CreateOSRMOneToManyResponse(mc, mc.CreateVectorOfStructs(costs))
              .Union());
      return make_msg(mc);
    });

    instance_->register_op("/osrm/route", [&](msg_ptr const& msg) {
      auto const req = motis_content(OSRMRouteRequest, msg);
      std::vector<latlng> coordinates =
          utl::to_vec(*req->coordinates(), [](auto* co) {
            return latlng{co->lat(), co->lng()};
          });
      std::vector<Cost> costs;
      costs.reserve(coordinates.size() - 1);
      auto const rrs_sz = coordinates.size() - 1;
      for (auto i = 0u; i < rrs_sz; i++) {
        costs.emplace_back(1000000 / rrs_sz, 1000000 / rrs_sz);
      }

      message_creator mc;
      mc.create_and_finish(
          MsgContent_OSRMRouteResponse,
          CreateOSRMRouteResponse(mc, mc.CreateVectorOfStructs(costs)).Union());
      return make_msg(mc);
    });
  }
};

TEST_F(rs_simple_itest, simple_with_ridesharing) {
  publish(make_no_msg("/osrm/initialized"));

  message_creator mc;

  Position rs_start{50, 6};
  Position rs_destination{51, 7};

  mc.create_and_finish(
      MsgContent_RidesharingCreate,
      CreateRidesharingCreate(mc, 12345, 11, 4, &rs_start, &rs_destination)
          .Union(),
      "/ridesharing/create");
  call(make_msg(mc));

  Position start{3, 3};
  Position destination{4, 4};
  std::string profile{"default"};

  mc.create_and_finish(
      MsgContent_RidesharingRequest,
      CreateRidesharingRequest(mc, &start, &destination, 10, 1, QUERYMODE_BOTH,
                               motis::ppr::CreateSearchOptions(
                                   mc, mc.CreateString(profile), 1000000))
          .Union(),
      "/ridesharing/edges");

  auto res = call(make_msg(mc));
  auto content = motis_content(RidesharingResponse, res);

  ASSERT_EQ(0, content->arrs()->size());
  ASSERT_EQ(0, content->deps()->size());
  ASSERT_EQ(0, content->direct_connections()->size());

  Position start1{50.8, 6.1};
  Position destination1{50.8, 6.1};
  mc.create_and_finish(MsgContent_RidesharingRequest,
                       CreateRidesharingRequest(
                           mc, &start1, &destination1, 10, 1, QUERYMODE_BOTH,
                           motis::ppr::CreateSearchOptions(
                               mc, mc.CreateString(profile), 1000000))
                           .Union(),
                       "/ridesharing/edges");

  // All (40 stations) match because of how the OSRM-Responses are mocked
  res = call(make_msg(mc));
  content = motis_content(RidesharingResponse, res);

  ASSERT_EQ(40, content->arrs()->size());
  ASSERT_EQ(40, content->deps()->size());
  ASSERT_EQ(1, content->direct_connections()->size());

  Position pick_up = {5, 5};
  Position drop_off = {6, 6};

  mc.create_and_finish(MsgContent_RidesharingBook,
                       CreateRidesharingBook(mc, 12345, 1, 7890, 3, 100000,
                                             &pick_up, 0, &drop_off, 0, 200)
                           .Union(),
                       "/ridesharing/book");
  call(make_msg(mc));

  mc.create_and_finish(
      MsgContent_RidesharingRequest,
      CreateRidesharingRequest(mc, &start1, &destination1, 0, 5, QUERYMODE_BOTH,
                               motis::ppr::CreateSearchOptions(
                                   mc, mc.CreateString(profile), 1000000))
          .Union(),
      "/ridesharing/edges");
  res = call(make_msg(mc));
  content = motis_content(RidesharingResponse, res);

  ASSERT_EQ(0, content->arrs()->size());
  ASSERT_EQ(0, content->deps()->size());
  ASSERT_EQ(0, content->direct_connections()->size());
}

}  // namespace motis::ridesharing
