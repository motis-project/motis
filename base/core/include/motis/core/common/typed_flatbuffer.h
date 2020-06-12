#pragma once

#include "flatbuffers/flatbuffers.h"

namespace motis {

template <typename T>
struct typed_flatbuffer {
  explicit typed_flatbuffer(FLATBUFFERS_NAMESPACE::FlatBufferBuilder&& fbb)
      : buffer_size_(fbb.GetSize()), buffer_(fbb.ReleaseBufferPointer()) {}

  typed_flatbuffer(size_t buffer_size,
                   FLATBUFFERS_NAMESPACE::unique_ptr_t buffer)
      : buffer_size_(buffer_size), buffer_(std::move(buffer)) {}

  explicit typed_flatbuffer(size_t buffer_size)
      : buffer_size_(buffer_size),
        buffer_(reinterpret_cast<uint8_t*>(operator new[](buffer_size)),
                std::default_delete<uint8_t[]>()) {}

  template <typename Buffer>
  typed_flatbuffer(size_t buffer_size,  // NOLINT (delegating member init)
                   Buffer* buffer)
      : typed_flatbuffer(buffer_size) {
    std::memcpy(buffer_.get(), buffer, buffer_size_);
  }

  explicit typed_flatbuffer(  // NOLINT (delegating member init)
      std::string const& s)
      : typed_flatbuffer(s.size(), s.data()) {}

  explicit typed_flatbuffer(  // NOLINT (delegating member init)
      std::string_view s)
      : typed_flatbuffer(s.size(), s.data()) {}

  typed_flatbuffer(typed_flatbuffer const&) = delete;
  typed_flatbuffer& operator=(typed_flatbuffer const&) = delete;

  typed_flatbuffer(typed_flatbuffer&&) = default;  // NOLINT
  typed_flatbuffer& operator=(typed_flatbuffer&&) = default;  // NOLINT

  virtual ~typed_flatbuffer() = default;

  uint8_t const* data() const { return buffer_.get(); }
  size_t size() const { return buffer_size_; }

  T* get() const {
    return FLATBUFFERS_NAMESPACE::GetMutableRoot<T>(buffer_.get());
  };

  std::string to_string() const {
    return {reinterpret_cast<char const*>(data()), size()};
  }

  size_t buffer_size_;
  FLATBUFFERS_NAMESPACE::unique_ptr_t buffer_;
};

}  // namespace motis
