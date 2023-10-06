#include "motis/transfers/platform/extract.h"

#include "motis/transfers/platform/from_osm.h"

namespace fs = std::filesystem;

namespace motis::transfers {

platforms extract_platforms_from_osm_file(fs::path const& osm_file_path) {
  auto osm_extractor = osm_platform_extractor(osm_file_path);

  for (auto const& filter_rule : filter_rule_descriptions) {
    osm_extractor.add_filter_rule(filter_rule.include_,
                                  filter_rule.key_matcher_,
                                  filter_rule.value_matcher_);
  }

  for (auto const& key : platform_name_keys) {
    osm_extractor.add_platform_name_tag_key(key);
  }

  return osm_extractor.get_platforms_identified_in_osm_file();
}

}  // namespace motis::transfers
