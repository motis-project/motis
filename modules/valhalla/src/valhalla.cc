#include "motis/valhalla/valhalla.h"

#include <filesystem>

#include "cista/reflection/comparable.h"

#include "baldr/rapidjson_utils.h"
#include "config.h"
#include "filesystem.h"
#include "midgard/logging.h"
#include "midgard/util.h"
#include "mjolnir/util.h"

#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"
#include "baldr/graphreader.h"
#include "loki/worker.h"
#include "sif/costfactory.h"
#include "sif/dynamiccost.h"
#include "thor/costmatrix.h"
#include "thor/optimizer.h"
#include "worker.h"

namespace mm = motis::module;
namespace fs = std::filesystem;
namespace v = ::valhalla;

namespace motis::valhalla {

boost::property_tree::ptree get_config(std::string const& tile_dir) {
  auto const config_json = fmt::format(R"({{
  "thor": {{
    "logging": {{
      "long_request": 110,
      "type": "std_out",
      "color": true
    }}
  }},
  "mjolnir": {{
    "tile_dir": "{}",
    "data_processing": {{
      "use_admin_db": false
    }}
  }},
  "loki": {{
    "actions": [
      "locate",
      "route",
      "height",
      "sources_to_targets",
      "optimized_route",
      "isochrone",
      "trace_route",
      "trace_attributes",
      "transit_available",
      "expansion",
      "centroid",
      "status"
    ],
    "use_connectivity": true,
    "service_defaults": {{
      "radius": 0,
      "minimum_reachability": 50,
      "search_cutoff": 35000,
      "node_snap_tolerance": 5,
      "street_side_tolerance": 5,
      "street_side_max_distance": 1000,
      "heading_tolerance": 60
    }},
    "logging": {{
      "type": "std_out",
      "color": true,
      "file_name": "path_to_some_file.log",
      "long_request": 100.0
    }},
    "service": {{
      "proxy": "ipc:///tmp/loki"
    }}
  }},
  "meili": {{
    "mode": "auto",
    "customizable": [
      "mode",
      "search_radius",
      "turn_penalty_factor",
      "gps_accuracy",
      "interpolation_distance",
      "sigma_z",
      "beta",
      "max_route_distance_factor",
      "max_route_time_factor"
    ],
    "verbose": false,
    "default": {{
      "sigma_z": 4.07,
      "gps_accuracy": 5.0,
      "beta": 3,
      "max_route_distance_factor": 5,
      "max_route_time_factor": 5,
      "max_search_radius": 100,
      "breakage_distance": 2000,
      "interpolation_distance": 10,
      "search_radius": 50,
      "geometry": false,
      "route": true,
      "turn_penalty_factor": 0
    }},
    "auto": {{
      "turn_penalty_factor": 200,
      "search_radius": 50
    }},
    "pedestrian": {{
      "turn_penalty_factor": 100,
      "search_radius": 50
    }},
    "bicycle": {{
      "turn_penalty_factor": 140
    }},
    "multimodal": {{
      "turn_penalty_factor": 70
    }},
    "logging": {{
      "type": "std_out",
      "color": true,
      "file_name": "path_to_some_file.log"
    }},
    "service": {{
      "proxy": "ipc:///tmp/meili"
    }},
    "grid": {{
      "size": 500,
      "cache_size": 100240
    }}
  }},
  "service_limits": {{
    "auto": {{
      "max_distance": 5000000.0,
      "max_locations": 20,
      "max_matrix_distance": 400000.0,
      "max_matrix_location_pairs": 2500
    }},
    "bus": {{
      "max_distance": 5000000.0,
      "max_locations": 50,
      "max_matrix_distance": 400000.0,
      "max_matrix_location_pairs": 2500
    }},
    "taxi": {{
      "max_distance": 5000000.0,
      "max_locations": 20,
      "max_matrix_distance": 400000.0,
      "max_matrix_location_pairs": 2500
    }},
    "pedestrian": {{
      "max_distance": 250000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500,
      "min_transit_walking_distance": 1,
      "max_transit_walking_distance": 10000
    }},
    "motor_scooter": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500
    }},
    "motorcycle": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500
    }},
    "bicycle": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500
    }},
    "multimodal": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 0.0,
      "max_matrix_location_pairs": 0
    }},
    "status": {{
      "allow_verbose": false
    }},
    "transit": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500
    }},
    "truck": {{
      "max_distance": 5000000.0,
      "max_locations": 20,
      "max_matrix_distance": 400000.0,
      "max_matrix_location_pairs": 2500
    }},
    "skadi": {{
      "max_shape": 750000,
      "min_resample": 10.0
    }},
    "isochrone": {{
      "max_contours": 4,
      "max_time_contour": 120,
      "max_distance": 25000.0,
      "max_locations": 1,
      "max_distance_contour": 200
    }},
    "trace": {{
      "max_alternates": 3,
      "max_alternates_shape": 100,
      "max_distance": 200000.0,
      "max_gps_accuracy": 100.0,
      "max_search_radius": 100.0,
      "max_shape": 16000
    }},
    "bikeshare": {{
      "max_distance": 500000.0,
      "max_locations": 50,
      "max_matrix_distance": 200000.0,
      "max_matrix_location_pairs": 2500
    }},
    "centroid": {{
      "max_distance": 200000.0,
      "max_locations": 5
    }},
    "max_exclude_locations": 50,
    "max_reachability": 100,
    "max_radius": 200,
    "max_timedep_distance": 500000,
    "max_alternates": 2,
    "max_exclude_polygons_length": 10000
    }}
}})",
                                       tile_dir);

  std::stringstream ss;
  ss << config_json;

  boost::property_tree::ptree pt;
  rapidjson::read_json(ss, pt);

  return pt;
}

struct import_state {
  CISTA_COMPARABLE()
  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

struct valhalla::impl {
  impl(boost::property_tree::ptree const& pt)
      : reader_{std::make_shared<v::baldr::GraphReader>(
            pt.get_child("mjolnir"))},
        loki_worker_{pt, reader_} {}
  std::shared_ptr<v::baldr::GraphReader> reader_;
  v::thor::CostMatrix matrix_;
  v::loki::loki_worker_t loki_worker_;
  v::sif::CostFactory factory_;
};

valhalla::valhalla() : module("Valhalla Street Router", "valhalla") {}

valhalla::~valhalla() noexcept = default;

void valhalla::init(mm::registry& reg) {
  auto const config = get_config(get_data_directory() / "valhalla");
  auto logging_subtree = config.get_child_optional("thor.logging");
  if (logging_subtree) {
    auto logging_config =
        v::midgard::ToMap<const boost::property_tree::ptree&,
                          std::unordered_map<std::string, std::string>>(
            logging_subtree.get());
    v::midgard::logging::Configure(logging_config);
  }
  impl_ = std::make_unique<impl>(config);
  reg.register_op("/valhalla",
                  [&](mm::msg_ptr const& msg) { return route(msg); }, {});
}

mm::msg_ptr valhalla::route(mm::msg_ptr const& msg) {
  using osrm::OSRMOneToManyRequest;
  namespace json = rapidjson;
  auto const req = motis_content(OSRMOneToManyRequest, msg);

  // Encode request.
  json::Document doc;
  doc.SetObject();

  auto const encode = [&](Position const* to) {
    auto coord = json::Value{json::kObjectType};
    coord.AddMember("lat", json::Value{to->lat()}, doc.GetAllocator());
    coord.AddMember("lon", json::Value{to->lng()}, doc.GetAllocator());
    return coord;
  };

  auto sources = json::Value{json::kArrayType};
  sources.PushBack(encode(req->one()), doc.GetAllocator());

  auto targets = json::Value{json::kArrayType};
  for (auto const& to : *req->many()) {
    targets.PushBack(encode(to), doc.GetAllocator());
  }

  doc.AddMember("sources", sources, doc.GetAllocator());
  doc.AddMember("targets", targets, doc.GetAllocator());
  doc.AddMember("costing", "pedestrian", doc.GetAllocator());

  // Decode request.
  v::Api request;
  from_json(doc, v::Options::sources_to_targets, request);
  auto& options = *request.mutable_options();

  // Get the costing method - pass the JSON configuration
  v::sif::TravelMode mode;
  auto mode_costing = impl_->factory_.CreateModeCosting(options, mode);

  // Find path locations (loki) for sources and targets
  impl_->loki_worker_.matrix(request);

  // Timing with CostMatrix
  impl_->matrix_.clear();
  auto const res = impl_->matrix_.SourceToTarget(
      options.sources(), options.targets(), *impl_->reader_, mode_costing, mode,
      4000000.0f);

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

void valhalla::import(mm::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "valhalla", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::OSMEvent;

        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        auto const osm_stem = fs::path{fs::path{osm->path()->str()}.stem()}
                                  .stem()
                                  .generic_string();

        auto const dir = get_data_directory() / "valhalla";
        fs::create_directories(dir);

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          auto const config = get_config(dir);
          v::mjolnir::build_tile_set(config, {osm->path()->str()});
          mm::write_ini(dir / "import.ini", state);
        }
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

}  // namespace motis::valhalla
