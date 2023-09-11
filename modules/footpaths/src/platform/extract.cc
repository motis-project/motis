#include "motis/footpaths/platform/extract.h"

#include "motis/footpaths/platform/from_osm.h"

namespace fs = std::filesystem;

namespace motis::footpaths {

platforms extract_platforms_from_osm_file(fs::path const& osm_file_path) {
  auto osm_extractor = osm_platform_extractor(osm_file_path);

  for (auto const& [result, key_matcher, value_matcher] :
       filter_rule_descriptions) {
    osm_extractor.add_filter_rule(result, key_matcher, value_matcher);
  }

  for (auto const& key : platform_name_keys) {
    osm_extractor.add_platform_name_tag_key(key);
  }

  return osm_extractor.get_platforms_identified_in_osm_file();
}

}  // namespace motis::footpaths
