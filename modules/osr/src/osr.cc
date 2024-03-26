#include "motis/osr/osr.h"

#include <filesystem>

#include "boost/thread/tss.hpp"

#include "utl/to_vec.h"

#include "cista/reflection/comparable.h"

#include "osr/extract.h"
#include "osr/lookup.h"
#include "osr/route.h"
#include "osr/ways.h"

#include "motis/core/common/logging.h"
#include "motis/core/conv/position_conv.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

namespace mm = motis::module;
namespace fs = std::filesystem;
namespace o = osr;

namespace motis::osr {

struct import_state {
  CISTA_COMPARABLE()

  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

struct osr::impl {
  std::unique_ptr<o::ways> w_;
  std::unique_ptr<o::lookup> l_;
  boost::thread_specific_ptr<o::routing_state> s_;
};

osr::osr() : module("Open Street Router", "osr") {}

osr::~osr() noexcept = default;

void osr::init(mm::registry& reg) {
  impl_ = std::make_unique<impl>();
  reg.register_op("/osrm/one_to_many",
                  [&](mm::msg_ptr const& msg) { return one_to_many(msg); }, {});
  reg.register_op("/osrm/table",
                  [&](mm::msg_ptr const& msg) { return table(msg); }, {});
  reg.register_op("/osrm/via", [&](mm::msg_ptr const& msg) { return via(msg); },
                  {});
  reg.register_op("/ppr/route",
                  [&](mm::msg_ptr const& msg) { return ppr(msg); }, {});
}

mm::msg_ptr osr::table(mm::msg_ptr const& msg) const {
  using osrm::OSRMManyToManyRequest;
  auto const req = motis_content(OSRMManyToManyRequest, msg);
  (void)req;
  return mm::make_success_msg();
}

mm::msg_ptr osr::one_to_many(mm::msg_ptr const& msg) const {
  using osrm::OSRMOneToManyRequest;
  auto const req = motis_content(OSRMOneToManyRequest, msg);
  (void)req;
  return mm::make_success_msg();
}

mm::msg_ptr osr::via(mm::msg_ptr const& msg) const {
  using osrm::OSRMViaRouteRequest;
  auto const req = motis_content(OSRMViaRouteRequest, msg);

  utl::verify(req->waypoints()->size() == 2U, "no via points supported");

  auto const from = from_fbs(req->waypoints()->Get(0));
  auto const to = from_fbs(req->waypoints()->Get(1));

  if (impl_->s_.get() == nullptr) {
    impl_->s_.reset(new o::routing_state{});
  }

  auto const result =
      o::route(*impl_->w_, *impl_->l_, from, to, 7200U, *impl_->s_,
               o::read_profile(req->profile()->view()));

  utl::verify(result.has_value(), "no path found from {} to {} with profile {}",
              from, to, req->profile()->view());

  auto doubles = std::vector<double>{};
  for (auto const& p : result->polyline_) {
    doubles.emplace_back(p.lat());
    doubles.emplace_back(p.lng());
  }
  mm::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_OSRMViaRouteResponse,
      osrm::CreateOSRMViaRouteResponse(
          fbb, static_cast<int>(result->time_),
          static_cast<double>(0) /* TODO */,
          CreatePolyline(fbb, fbb.CreateVector(doubles.data(), doubles.size())))
          .Union());
  return make_msg(fbb);
}

mm::msg_ptr osr::ppr(mm::msg_ptr const& msg) const {
  using osrm::OSRMOneToManyResponse;
  using osrm::OSRMViaRouteResponse;
  using ppr::FootRoutingRequest;
  using ppr::FootRoutingResponse;

  auto const req = motis_content(FootRoutingRequest, msg);
  mm::message_creator fbb;
  if (req->include_path()) {
    fbb.create_and_finish(
        MsgContent_FootRoutingResponse,
        ppr::CreateFootRoutingResponse(
            fbb,
            fbb.CreateVector(utl::to_vec(
                *req->destinations(),
                [&](Position const* dest) {
                  mm::message_creator req_fbb;
                  auto const from_to = std::array<Position, 2>{
                      // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
                      req->search_direction() == SearchDir_Forward
                          ? *req->start()
                          : *dest,
                      // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
                      req->search_direction() == SearchDir_Forward
                          ? *dest
                          : *req->start()};
                  req_fbb.create_and_finish(
                      MsgContent_OSRMViaRouteRequest,
                      osrm::CreateOSRMViaRouteRequest(
                          req_fbb, req_fbb.CreateString("foot"),
                          req_fbb.CreateVectorOfStructs(from_to.data(), 2U))
                          .Union());
                  auto const res_msg = via(make_msg(req_fbb));
                  auto const res = motis_content(OSRMViaRouteResponse, res_msg);
                  return ppr::CreateRoutes(
                      fbb,
                      fbb.CreateVector(std::vector{ppr::CreateRoute(
                          fbb, res->distance(), res->time(), res->time(), 0.0,
                          0U, 0.0, 0.0, req->start(), dest,
                          fbb.CreateVector(
                              std::vector<
                                  flatbuffers::Offset<ppr::RouteStep>>{}),
                          fbb.CreateVector(
                              std::vector<flatbuffers::Offset<ppr::Edge>>{}),
                          motis_copy_table(Polyline, fbb, res->polyline()), 0,
                          0)}));
                })))
            .Union());
  } else {
    mm::message_creator req_fbb;
    auto const start = *req->start();
    auto const dests =
        utl::to_vec(*req->destinations(), [](Position const* dest) {
          return Position{dest->lat(), dest->lng()};
        });
    req_fbb.create_and_finish(
        MsgContent_OSRMOneToManyRequest,
        osrm::CreateOSRMOneToManyRequest(
            req_fbb, req_fbb.CreateString("foot"), req->search_direction(),
            &start, req_fbb.CreateVectorOfStructs(dests.data(), dests.size()))
            .Union(),
        "/osrm/one_to_many");
    auto const res_msg = one_to_many(make_msg(req_fbb));
    auto const res = motis_content(OSRMOneToManyResponse, res_msg);
    fbb.create_and_finish(
        MsgContent_FootRoutingResponse,
        ppr::CreateFootRoutingResponse(
            fbb,
            fbb.CreateVector(utl::to_vec(
                *res->costs(),
                [&, i = 0](osrm::Cost const* cost) mutable {
                  auto const vec = std::vector{ppr::CreateRoute(
                      fbb, cost->distance(), cost->duration(), cost->duration(),
                      0.0, 0U, 0.0, 0.0, req->start(),
                      req->destinations()->Get(i++),
                      fbb.CreateVector(
                          std::vector<flatbuffers::Offset<ppr::RouteStep>>{}),
                      fbb.CreateVector(
                          std::vector<flatbuffers::Offset<ppr::Edge>>{}),
                      CreatePolyline(fbb,
                                     fbb.CreateVector(std::vector<double>{})))};
                  return ppr::CreateRoutes(fbb, fbb.CreateVector(vec));
                })))
            .Union());
  }
  return make_msg(fbb);
}

void osr::import(mm::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "osr", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::OSMEvent;

        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        auto const dir = get_data_directory() / "osr";
        fs::create_directories(dir);

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          o::extract(state.path_.val(), dir);
          mm::write_ini(dir / "import.ini", state);
        }

        impl_->w_ =
            std::make_unique<o::ways>(dir, cista::mmap::protection::READ);
        impl_->l_ = std::make_unique<o::lookup>(*impl_->w_);
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

}  // namespace motis::osr
