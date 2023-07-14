#include "motis/valhalla/config.h"

#include <sstream>

#include "valhalla/baldr/rapidjson_utils.h"

#include "fmt/core.h"

namespace motis::valhalla {

boost::property_tree::ptree get_config(std::string const& tile_dir) {
  auto const config_json = fmt::format(R"({{
  "thor": {{
  }},
  "mjolnir": {{
    "tile_dir": "{}",
    "data_processing": {{
      "use_admin_db": false
    }},
    "logging": {{
      "type": "std_out"
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

}  // namespace motis::valhalla