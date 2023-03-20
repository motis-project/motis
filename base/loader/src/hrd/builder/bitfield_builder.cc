#include "motis/loader/hrd/builder/bitfield_builder.h"

#include "utl/get_or_create.h"
#include "utl/verify.h"

#include "motis/loader/util.h"

using namespace flatbuffers64;

namespace motis::loader::hrd {

bitfield_builder::bitfield_builder(std::map<int, bitfield> hrd_bitfields)
    : hrd_bitfields_(std::move(hrd_bitfields)) {
  // TODO (Felix Guendling) what happens if there is a service such traffic days
  bitfield alternating_bits;
  for (size_t i = 0; i < alternating_bits.size(); ++i) {
    alternating_bits.set(i, (i % 2) == 0);
  }
}

Offset<String> bitfield_builder::get_or_create_bitfield(
    int bitfield_num, flatbuffers64::FlatBufferBuilder& fbb) {
  auto lookup_it = fbs_bf_lookup_.find(bitfield_num);
  if (lookup_it != end(fbs_bf_lookup_)) {
    return lookup_it->second;
  }

  auto hrd_bitfields_it = hrd_bitfields_.find(bitfield_num);
  utl::verify(hrd_bitfields_it != end(hrd_bitfields_),
              "bitfield with bitfield number {} not found\n", bitfield_num);
  return get_or_create_bitfield(hrd_bitfields_it->second, fbb, bitfield_num);
}

Offset<String> bitfield_builder::get_or_create_bitfield(
    bitfield const& b, flatbuffers64::FlatBufferBuilder& fbb,
    int bitfield_num) {
  return utl::get_or_create(fbs_bitfields_, b, [&]() {
    auto const serialized = fbb.CreateString(serialize_bitset<BIT_COUNT>(b));
    if (bitfield_num != kNoBitfield) {
      fbs_bf_lookup_.emplace(bitfield_num, serialized);
    }
    return serialized;
  });
}

}  // namespace motis::loader::hrd
