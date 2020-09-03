#pragma once

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reflection.h"

#ifdef GetMessage
#undef GetMessage
#endif

#include "motis/protocol/Message_generated.h"

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/module/error.h"

namespace motis::module {

constexpr auto const DEFAULT_FBS_MAX_DEPTH = 64;
constexpr auto const DEFAULT_FBS_MAX_TABLES = 1'000'000;

class message_creator : public flatbuffers::FlatBufferBuilder {
public:
  void create_and_finish(
      MsgContent type, flatbuffers::Offset<void> content,
      std::string const& target = "",
      DestinationType destination_type = DestinationType_Module,
      int const id = 1) {
    Finish(CreateMessage(
        *this, CreateDestination(*this, destination_type, CreateString(target)),
        type, content, id));
  }
};

struct message : public typed_flatbuffer<Message> {
  message() : typed_flatbuffer(0, nullptr) {}
  message(size_t len, flatbuffers::unique_ptr_t mem)
      : typed_flatbuffer(len, std::move(mem)) {}
  message(size_t len, void const* ptr) : typed_flatbuffer(len, ptr) {}
  explicit message(size_t len) : typed_flatbuffer(len) {}

  int id() const { return get()->id(); }

  std::string to_json(bool compact = false) const;

  static reflection::Schema const& get_schema();
  static reflection::Object const* get_objectref(char const* name);
  static std::pair<const char**, size_t> get_fbs_definitions();
};

using msg_ptr = std::shared_ptr<message>;

msg_ptr make_msg(std::string const& json, bool fix = false,
                 std::size_t fbs_max_depth = DEFAULT_FBS_MAX_DEPTH,
                 std::size_t fbs_max_tables = DEFAULT_FBS_MAX_TABLES);
msg_ptr make_msg(message_creator& builder);
msg_ptr make_msg(void const* buf, size_t len,
                 std::size_t fbs_max_depth = DEFAULT_FBS_MAX_DEPTH,
                 std::size_t fbs_max_tables = DEFAULT_FBS_MAX_TABLES);

msg_ptr make_no_msg(std::string const& target = "", int id = 1);
msg_ptr make_success_msg(std::string const& target = "", int id = 1);
msg_ptr make_error_msg(std::error_code const&, int id = 1);
msg_ptr make_unknown_error_msg(std::string const& reason, int id = 1);

template <typename T>
inline T const* motis_content_impl(msg_ptr const& msg,
                                   MsgContent content_type) {
  if (msg == nullptr) {
    assert(false);
    throw std::system_error(error::null_message_content_access);
  } else if (msg->get()->content_type() != content_type) {
    throw std::system_error(error::unexpected_message_type);
  }
  return reinterpret_cast<T const*>(msg->get()->content());
}

#define motis_content(content_type, msg)               \
  motis::module::motis_content_impl<content_type>(msg, \
                                                  MsgContent_##content_type)

template <typename TableType>
inline flatbuffers::Offset<TableType> motis_copy_table_impl(  // NOLINT
    flatbuffers::FlatBufferBuilder& fbb, char const* table_name,
    void const* src) {
  return flatbuffers::Offset<TableType>(
      flatbuffers::CopyTable(fbb, message::get_schema(),
                             *message::get_objectref(table_name),
                             *(flatbuffers::Table*)src)  // NOLINT
          .o);
}

#define motis_copy_table(table_type, target_builder, src)          \
  motis::module::motis_copy_table_impl<table_type>(target_builder, \
                                                   #table_type, src)

}  // namespace motis::module
