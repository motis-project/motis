#include "motis/paxforecast/behavior/parser.h"

#include <random>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "utl/verify.h"

#include "motis/json/json.h"

#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

using namespace motis::json;

namespace motis::paxforecast::behavior {

std::normal_distribution<float> parse_normal_distribution(
    rapidjson::Value const& params) {
  return std::normal_distribution{
      static_cast<float>(get_double(params, "mean")),
      static_cast<float>(get_double(params, "stddev"))};
}

probabilistic::passenger_behavior parse_passenger_behavior(
    std::string const& json, unsigned const default_sample_count,
    bool const default_best_only) {
  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  utl::verify(!doc.HasParseError(), "bad json: {} at offset {}",
              rapidjson::GetParseError_En(doc.GetParseError()),
              doc.GetErrorOffset());

  utl::verify(doc.IsObject(), "no root object");

  auto sample_count = default_sample_count;
  auto best_only = default_best_only;

  auto const model = get_str(doc, "model");
  utl::verify(model == "probabilistic", "paxforecast: unsupported model type");

  if (has_key(doc, "best_only")) {
    best_only = get_bool(doc, "best_only");
  }

  if (has_key(doc, "sample_count")) {
    sample_count = get_uint(doc, "sample_count");
  }

  auto const& params = get_obj(doc, "parameters");
  return probabilistic::passenger_behavior{
      sample_count,
      best_only,
      parse_normal_distribution(get_obj(params, "transfer")),
      parse_normal_distribution(get_obj(params, "original")),
      parse_normal_distribution(get_obj(params, "recommended")),
      parse_normal_distribution(get_obj(params, "load_info")),
      parse_normal_distribution(get_obj(params, "load_info_low")),
      parse_normal_distribution(get_obj(params, "load_info_no_seats")),
      parse_normal_distribution(get_obj(params, "load_info_full")),
      parse_normal_distribution(get_obj(params, "random"))};
}

}  // namespace motis::paxforecast::behavior
