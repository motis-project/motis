#include "motis/valhalla/valhalla.h"

#include <filesystem>

#include "cista/reflection/comparable.h"

#include "baldr/attributes_controller.h"
#include "baldr/graphreader.h"
#include "baldr/rapidjson_utils.h"
#include "config.h"
#include "filesystem.h"
#include "loki/worker.h"
#include "midgard/util.h"
#include "mjolnir/util.h"
#include "odin/directionsbuilder.h"
#include "sif/costfactory.h"
#include "sif/dynamiccost.h"
#include "thor/bidirectional_astar.h"
#include "thor/costmatrix.h"
#include "thor/optimizer.h"
#include "thor/triplegbuilder.h"
#include "worker.h"

#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"
#include "motis/valhalla/config.h"

namespace mm = motis::module;
namespace fs = std::filesystem;
namespace v = valhalla;
namespace j = rapidjson;

namespace motis::valhalla {

struct import_state {
  CISTA_COMPARABLE()
  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

struct valhalla::impl {
  explicit impl(boost::property_tree::ptree const& pt)
      : reader_{std::make_shared<v::baldr::GraphReader>(
            pt.get_child("mjolnir"))},
        loki_worker_{pt, reader_},
        bd_{pt.get_child("thor")},
        formatter_{pt} {}
  std::shared_ptr<v::baldr::GraphReader> reader_;
  v::thor::CostMatrix matrix_;
  v::loki::loki_worker_t loki_worker_;
  v::sif::CostFactory factory_;
  v::thor::BidirectionalAStar bd_;
  v::odin::MarkupFormatter formatter_;
};

valhalla::valhalla() : module("Valhalla Street Router", "valhalla") {}

valhalla::~valhalla() noexcept = default;

void valhalla::init(mm::registry& reg) {
  auto const config =
      get_config((get_data_directory() / "valhalla").generic_string());
  impl_ = std::make_unique<impl>(config);
  reg.register_op("/osrm/one_to_many",
                  [&](mm::msg_ptr const& msg) { return one_to_many(msg); }, {});
  reg.register_op("/osrm/table",
                  [&](mm::msg_ptr const& msg) { return table(msg); }, {});
  reg.register_op("/osrm/via", [&](mm::msg_ptr const& msg) { return via(msg); },
                  {});
}

std::string_view translate_mode(std::string_view s) {
  switch (cista::hash(s)) {
    case cista::hash("foot"): return "pedestrian";
    case cista::hash("bike"): return "bicycle";
    case cista::hash("car"): return "auto";
    default: return "pedestrian";
  }
}

j::Value encode_pos(Position const* to, j::Document& doc) {
  auto coord = j::Value{j::kObjectType};
  coord.AddMember("lat", j::Value{to->lat()}, doc.GetAllocator());
  coord.AddMember("lon", j::Value{to->lng()}, doc.GetAllocator());
  return coord;
}

void encode_request(osrm::OSRMManyToManyRequest const* req, j::Document& doc) {
  doc.SetObject();

  auto sources = j::Value{j::kArrayType};
  for (auto const& from : *req->from()) {
    sources.PushBack(encode_pos(from, doc), doc.GetAllocator());
  }

  auto targets = j::Value{j::kArrayType};
  for (auto const& to : *req->to()) {
    targets.PushBack(encode_pos(to, doc), doc.GetAllocator());
  }

  doc.AddMember("sources", sources, doc.GetAllocator());
  doc.AddMember("targets", targets, doc.GetAllocator());

  auto const mode_str = translate_mode(req->profile()->view());
  doc.AddMember("costing", j::StringRef(mode_str.data(), mode_str.size()),
                doc.GetAllocator());
}

void encode_request(osrm::OSRMOneToManyRequest const* req, j::Document& doc) {
  doc.SetObject();

  auto sources = j::Value{j::kArrayType};
  sources.PushBack(encode_pos(req->one(), doc), doc.GetAllocator());

  auto targets = j::Value{j::kArrayType};
  for (auto const& to : *req->many()) {
    targets.PushBack(encode_pos(to, doc), doc.GetAllocator());
  }

  doc.AddMember("sources",
                req->direction() == SearchDir_Forward ? sources : targets,
                doc.GetAllocator());
  doc.AddMember("targets",
                req->direction() == SearchDir_Forward ? targets : sources,
                doc.GetAllocator());

  auto const mode_str = translate_mode(req->profile()->view());
  doc.AddMember("costing", j::StringRef(mode_str.data(), mode_str.size()),
                doc.GetAllocator());
}

void encode_request(osrm::OSRMViaRouteRequest const* req, j::Document& doc) {
  doc.SetObject();

  auto locations = j::Value{j::kArrayType};
  for (auto const& to : *req->waypoints()) {
    locations.PushBack(encode_pos(to, doc), doc.GetAllocator());
  }

  doc.AddMember("locations", locations, doc.GetAllocator());

  auto const mode_str = translate_mode(req->profile()->view());
  doc.AddMember("costing", j::StringRef(mode_str.data(), mode_str.size()),
                doc.GetAllocator());
}

template <typename Req>
mm::msg_ptr sources_to_targets(Req const* req, valhalla::impl* impl_) {
  // Encode OSRMManyToManyRequest as valhalla request.
  auto doc = j::Document{};
  encode_request(req, doc);

  // Decode request.
  v::Api request;
  v::from_json(doc, v::Options::sources_to_targets, request);
  auto& options = *request.mutable_options();

  // Get the costing method.
  auto mode = v::sif::TravelMode::kMaxTravelMode;
  auto const mode_costing = impl_->factory_.CreateModeCosting(options, mode);

  // Find path locations (loki) for sources and targets.
  impl_->loki_worker_.matrix(request);

  // Run matrix algorithm.
  impl_->matrix_.clear();
  auto const res = impl_->matrix_.SourceToTarget(
      options.sources(), options.targets(), *impl_->reader_, mode_costing, mode,
      4000000.0F);

  // Encode OSRM response.
  mm::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_OSRMOneToManyResponse,
      CreateOSRMOneToManyResponse(
          fbb, fbb.CreateVectorOfStructs(utl::to_vec(
                   res,
                   [](v::thor::TimeDistance const& td) {
                     return motis::osrm::Cost{1.0 * td.time, 1.0 * td.dist};
                   })))
          .Union());
  return make_msg(fbb);
}

mm::msg_ptr valhalla::table(mm::msg_ptr const& msg) const {
  using osrm::OSRMManyToManyRequest;
  auto const req = motis_content(OSRMManyToManyRequest, msg);
  return sources_to_targets(req, impl_.get());
}

mm::msg_ptr valhalla::one_to_many(mm::msg_ptr const& msg) const {
  using osrm::OSRMOneToManyRequest;
  auto const req = motis_content(OSRMOneToManyRequest, msg);
  return sources_to_targets(req, impl_.get());
}

mm::msg_ptr valhalla::via(mm::msg_ptr const& msg) const {
  using osrm::OSRMViaRouteRequest;
  auto const req = motis_content(OSRMViaRouteRequest, msg);

  // Encode OSRMViaRouteRequest as valhalla request.
  auto doc = j::Document{};
  encode_request(req, doc);

  // Decode request.
  v::Api request;
  v::from_json(doc, v::Options::route, request);
  auto const& options = *request.mutable_options();

  // Get the costing method.
  auto mode = v::sif::TravelMode::kMaxTravelMode;
  auto const& mode_costing = impl_->factory_.CreateModeCosting(options, mode);

  // Find path locations (loki) for sources and targets.
  impl_->loki_worker_.route(request);

  // Compute paths.
  for (uint32_t i = 0; i < options.locations().size() - 1U; i++) {
    auto origin = options.locations(i);
    auto dest = options.locations(i + 1);

    auto paths = impl_->bd_.GetBestPath(origin, dest, *impl_->reader_,
                                        mode_costing, mode, request.options());
    auto cost = mode_costing[static_cast<uint32_t>(mode)];

    // If bidirectional A*, disable use of destination only edges on the first
    // pass. If there is a failure, we allow them on the second pass.
    cost->set_allow_destination_only(false);
    cost->set_pass(0);
    if (paths.empty() ||
        (mode == v::sif::TravelMode::kPedestrian && impl_->bd_.has_ferry())) {
      if (cost->AllowMultiPass()) {
        cost->set_pass(1);
        impl_->bd_.Clear();
        cost->RelaxHierarchyLimits(true);
        cost->set_allow_destination_only(true);
        paths = impl_->bd_.GetBestPath(origin, dest, *impl_->reader_,
                                       mode_costing, mode, request.options());
      }
    }
    if (paths.empty()) {
      return nullptr;
    }

    // Form trip path
    auto const controller = AttributesController{};
    auto const& pathedges = paths.front();
    auto& trip_path =
        *request.mutable_trip()->mutable_routes()->Add()->mutable_legs()->Add();
    v::thor::TripLegBuilder::Build(request.options(), controller,
                                   *impl_->reader_, mode_costing,
                                   pathedges.begin(), pathedges.end(), origin,
                                   dest, trip_path, {impl_->bd_.name()});

    impl_->bd_.Clear();
  }

  // Extract result.
  auto polyline = std::vector<PointLL>{};
  for (auto const& l : request.trip().routes(0).legs()) {
    auto const points = v::midgard::decode<std::vector<PointLL>>(
        l.shape().c_str(), l.shape().length());
    polyline.insert(end(polyline), begin(points), end(points));
  }

  // Get totals.
  v::odin::DirectionsBuilder::Build(request, impl_->formatter_);
  double total_time = 0.0;
  float total_distance = 0.0F;
  for (auto const& l : request.directions().routes(0).legs()) {
    total_time += l.summary().time();
    total_distance = l.summary().length();
  }

  // Encode OSRM response.
  std::vector<double> doubles;
  for (auto const& p : polyline) {
    doubles.emplace_back(p.lat());
    doubles.emplace_back(p.lng());
  }
  mm::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_OSRMViaRouteResponse,
      osrm::CreateOSRMViaRouteResponse(
          fbb, static_cast<int>(total_time),
          static_cast<double>(total_distance),
          CreatePolyline(fbb, fbb.CreateVector(doubles.data(), doubles.size())))
          .Union());
  return make_msg(fbb);
}

mm::msg_ptr valhalla::ppr(mm::msg_ptr const& msg) const {
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
                      req->search_direction() == SearchDir_Forward
                          ? *req->start()
                          : *dest,
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
    req_fbb.create_and_finish(MsgContent_OSRMOneToManyRequest,
                              osrm::CreateOSRMOneToManyRequest(
                                  fbb, fbb.CreateString("foot"),
                                  req->search_direction(), req->start(),
                                  fbb.CreateVector(req->destinations()->data(),
                                                   req->destinations()->size()))
                                  .Union());
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

void valhalla::import(mm::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "valhalla", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::OSMEvent;

        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        auto const dir = get_data_directory() / "valhalla";
        fs::create_directories(dir);

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          auto const config = get_config(dir.generic_string());
          v::mjolnir::build_tile_set(config, {osm->path()->str()});
          mm::write_ini(dir / "import.ini", state);
        }
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

}  // namespace motis::valhalla
