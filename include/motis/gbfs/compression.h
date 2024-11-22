#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "cista/containers/bitvec.h"

#include "utl/verify.h"

#include "lz4.h"

#include "motis/gbfs/data.h"

namespace motis::gbfs {

template <typename Vec, typename Key = typename Vec::size_type>
inline compressed_bitvec compress_bitvec(
    cista::basic_bitvec<Vec, Key> const& bv) {
  auto const* original_data = reinterpret_cast<char const*>(bv.blocks_.data());
  auto const original_bytes =
      static_cast<int>(bv.blocks_.size() *
                       sizeof(typename cista::basic_bitvec<Vec, Key>::block_t));
  auto const max_compressed_size = LZ4_compressBound(original_bytes);

  auto cbv = compressed_bitvec{
      .data_ =
          std::unique_ptr<char[], compressed_bitvec::free_deleter>{
              static_cast<char*>(
                  std::malloc(static_cast<std::size_t>(max_compressed_size)))},
      .original_bytes_ = original_bytes,
      .bitvec_size_ = bv.size_};
  utl::verify(cbv.data_ != nullptr,
              "could not allocate memory for compressed bitvec");

  cbv.compressed_bytes_ = LZ4_compress_default(
      original_data, cbv.data_.get(), original_bytes, max_compressed_size);
  utl::verify(cbv.compressed_bytes_ > 0, "could not compress bitvec");

  if (auto* compressed = std::realloc(
          cbv.data_.get(), static_cast<std::size_t>(cbv.compressed_bytes_));
      compressed != nullptr) {
    cbv.data_.release();
    cbv.data_.reset(static_cast<char*>(compressed));
  }
  return cbv;
}

template <typename Vec, typename Key = typename Vec::size_type>
inline void decompress_bitvec(compressed_bitvec const& cbv,
                              cista::basic_bitvec<Vec, Key>& bv) {
  bv.resize(static_cast<typename cista::basic_bitvec<Vec, Key>::size_type>(
      cbv.bitvec_size_));
  auto const decompressed_bytes = LZ4_decompress_safe(
      cbv.data_.get(), reinterpret_cast<char*>(bv.blocks_.data()),
      cbv.compressed_bytes_,
      static_cast<int>(
          bv.blocks_.size() *
          sizeof(typename cista::basic_bitvec<Vec, Key>::block_t)));
  utl::verify(decompressed_bytes == cbv.original_bytes_,
              "could not decompress bitvec");
}

}  // namespace motis::gbfs
