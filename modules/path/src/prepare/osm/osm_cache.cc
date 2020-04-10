#include "motis/path/prepare/osm/osm_cache.h"

#include <memory>

#include "utl/parser/file.h"
#include "utl/to_vec.h"

#include "motis/path/fbs/OsmCache_generated.h"

using namespace flatbuffers;

namespace motis::path {

std::vector<std::vector<osm_way>> load_osm_ways(std::string const& filename) {
  auto const buf = utl::file{filename.c_str(), "r"}.content();
  return utl::to_vec(
      *GetOsmCache(buf.buf_)->components(), [](auto const& component) {
        return utl::to_vec(*component->ways(), [](auto const& way) {
          return osm_way{
              way->from(), way->to(), utl::to_vec(*way->ids()),
              osm_path{utl::to_vec(*way->polyline(),
                                   [](auto const& pos) {
                                     return geo::latlng{pos->lat(), pos->lng()};
                                   }),
                       utl::to_vec(*way->osm_node_ids())},
              way->oneway()};
        });
      });
}

void store_osm_ways(std::string const& filename,
                    std::vector<std::vector<osm_way>> const& components) {
  FlatBufferBuilder fbb;

  auto const fbs_components =
      utl::to_vec(components, [&](auto const& component) {
        return CreateOsmComponent(
            fbb, fbb.CreateVector(utl::to_vec(component, [&](auto const& way) {
              return CreateOsmWay(
                  fbb, way.from_, way.to_, fbb.CreateVector(way.ids_),
                  fbb.CreateVectorOfStructs(
                      utl::to_vec(way.path_.polyline_,
                                  [](auto const& pos) {
                                    return Position(pos.lat_, pos.lng_);
                                  })),
                  fbb.CreateVector(way.path_.osm_node_ids_), way.oneway_);
            })));
      });

  fbb.Finish(CreateOsmCache(fbb, fbb.CreateVector(fbs_components)));
  utl::file{filename.c_str(), "w+"}.write(fbb.GetBufferPointer(),
                                          fbb.GetSize());
}

}  // namespace motis::path
