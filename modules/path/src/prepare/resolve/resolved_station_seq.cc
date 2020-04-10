#include "motis/path/prepare/resolve/resolved_station_seq.h"

#include <utl/to_vec.h>

#include "utl/parallel_for.h"
#include "utl/parser/file.h"
#include "utl/parser/mmap_reader.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/fixed/io/deserialize.h"
#include "tiles/fixed/io/serialize.h"

#include "motis/core/common/logging.h"

#include "motis/path/fbs/ResolvedStationSeqCache_generated.h"

using namespace flatbuffers;
namespace ml = motis::logging;

namespace motis::path {

void write_to_fbs(std::vector<resolved_station_seq> const& sequences,
                  std::string const& fname) {
#ifdef _MSC_VER
  FILE* file = nullptr;
  fopen_s(&file, fname.c_str(), "w+");
#else
  FILE* file = std::fopen(fname.c_str(), "w+e");
#endif
  utl::verify(file != nullptr, "cannot open file");

  constexpr size_t const kBatchSize = 10000UL;
  for (size_t begin = 0UL; begin < sequences.size(); begin += kBatchSize) {
    auto end = std::min(begin + kBatchSize, sequences.size());

    FlatBufferBuilder fbb;
    std::vector<Offset<InternalPathSeqResponse>> vec;
    for (auto i = begin; i < end; ++i) {
      auto const& seq = sequences[i];

      auto const fbs_stations =
          utl::to_vec(seq.station_ids_,
                      [&](auto const& id) { return fbb.CreateString(id); });

      auto fbs_segments = utl::to_vec(seq.paths_, [&](auto const& path) {
        tiles::fixed_polyline polyline;
        polyline.emplace_back();
        polyline.back().reserve(path.polyline_.size());
        for (auto const& pos : path.polyline_) {
          polyline.back().emplace_back(tiles::latlng_to_fixed(pos));
        }

        return CreateInternalSegment(
            fbb, fbb.CreateString(tiles::serialize(polyline)),
            fbb.CreateString(""), fbb.CreateVector(path.osm_node_ids_));
      });

      std::vector<Offset<InternalPathSourceInfo>> fbs_info;
      for (auto const& info : seq.sequence_infos_) {
        fbs_info.push_back(
            CreateInternalPathSourceInfo(fbb, info.idx_, info.from_, info.to_,
                                         fbb.CreateString(info.type_)));
      }

      vec.emplace_back(CreateInternalPathSeqResponse(
          fbb, fbb.CreateVector(fbs_stations), fbb.CreateVector(seq.classes_),
          fbb.CreateVector(fbs_segments), fbb.CreateVector(fbs_info)));
    }

    fbb.Finish(CreateResolvedStationSeqCache(fbb, fbb.CreateVector(vec)));

    uint32_t size = fbb.GetSize();
    auto const w_size = std::fwrite(reinterpret_cast<void const*>(&size),
                                    sizeof(uint32_t), 1, file);
    utl::verify(w_size == 1, "size write failed");

    auto const w_buf = std::fwrite(
        reinterpret_cast<void const*>(fbb.GetBufferPointer()), 1, size, file);
    utl::verify(w_buf == size, "buf write failed");
  }

  uint32_t terminal = 0;
  auto const w_terminal = std::fwrite(reinterpret_cast<void const*>(&terminal),
                                      sizeof(uint32_t), 1, file);
  utl::verify(w_terminal == 1, "terminal write failed");

  std::fclose(file);
}

std::vector<resolved_station_seq> read_from_fbs(std::string const& fname) {
  ml::scoped_timer t{"read_from_fbs"};
  auto mmap = utl::mmap_reader::memory_map(fname.c_str());

  struct batch {
    size_t buffer_offset_;
    size_t vector_offset_;
  };
  std::vector<batch> batches;

  size_t buffer_offset = 0;
  size_t vector_offset = 0;
  while (true) {
    uint32_t size;
    std::memcpy(&size, mmap.ptr() + buffer_offset, sizeof(uint32_t));
    buffer_offset += sizeof(uint32_t);

    if (size == 0) {
      break;
    }

    batches.push_back({buffer_offset, vector_offset});

    auto cache = GetResolvedStationSeqCache(mmap.ptr() + buffer_offset);
    vector_offset += cache->sequences()->size();
    buffer_offset += size;
  }

  utl::verify(!batches.empty(), "nothing to read?");

  std::vector<resolved_station_seq> result(vector_offset);
  utl::parallel_for(batches, [&](auto const& batch) {
    auto cache = GetResolvedStationSeqCache(mmap.ptr() + batch.buffer_offset_);
    for (auto i = 0u; i < cache->sequences()->size(); ++i) {
      auto const* cached_seq = cache->sequences()->Get(i);

      resolved_station_seq seq;
      seq.station_ids_ = utl::to_vec(*cached_seq->station_ids(),
                                     [](auto const* id) { return id->str(); });
      seq.classes_ = std::vector<uint32_t>(std::begin(*cached_seq->classes()),
                                           std::end(*cached_seq->classes()));
      seq.paths_ = utl::to_vec(*cached_seq->segments(), [](auto const* seg) {
        osm_path path;

        auto geometry = tiles::deserialize(seg->coords()->str());
        auto const& polyline = mpark::get<tiles::fixed_polyline>(geometry);
        utl::verify(polyline.size() == 1, "invalid polyline");
        path.polyline_ = utl::to_vec(polyline.front(), [](auto const& c) {
          return tiles::fixed_to_latlng(c);
        });
        path.osm_node_ids_ = std::vector<int64_t>(
            std::begin(*seg->osm_node_ids()), std::end(*seg->osm_node_ids()));
        path.verify_path();
        return path;
      });

      seq.sequence_infos_ =
          utl::to_vec(*cached_seq->infos(), [](auto const* info) {
            return sequence_info(info->segment_idx(), info->from_idx(),
                                 info->to_idx(), info->type()->str());
          });

      result.at(batch.vector_offset_ + i) = std::move(seq);
    }
  });

  std::cout << result.size() << std::endl;

  return result;
}

}  // namespace motis::path
