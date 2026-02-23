#include "motis/elevators/parse_elevator_id_osm_mapping.h"

#include "utl/parser/csv_range.h"

namespace motis {

elevator_id_osm_mapping_t parse_elevator_id_osm_mapping(std::string_view s) {
  struct row {
    utl::csv_col<utl::cstr, UTL_NAME("dhid")> dhid_;
    utl::csv_col<utl::cstr, UTL_NAME("diid")> diid_;
    utl::csv_col<utl::cstr, UTL_NAME("osm_kind")> osm_kind_;
    utl::csv_col<std::uint64_t, UTL_NAME("osm_id")> osm_id_;
  };
  auto map = elevator_id_osm_mapping_t{};
  utl::for_each_row<row>(s, [&](row const& r) {
    map.emplace(*r.osm_id_, std::string{r.diid_->view()});
  });
  return map;
}

elevator_id_osm_mapping_t parse_elevator_id_osm_mapping(
    std::filesystem::path const& p) {
  return parse_elevator_id_osm_mapping(
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ}
          .view());
}

}  // namespace motis
