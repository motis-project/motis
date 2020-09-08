#include "motis/module/message.h"

#include <cstring>
#include <stdexcept>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "motis/core/common/logging.h"
#include "motis/module/error.h"
#include "motis/module/fix_json.h"
#include "motis/protocol/resources.h"

#undef GetMessage

using namespace flatbuffers;

namespace motis::module {

std::unique_ptr<Parser> init_parser(bool compact = false) {
  auto parser = std::make_unique<Parser>();
  parser->opts.strict_json = true;
  parser->opts.skip_unexpected_fields_in_json = true;
  if (compact) {
    parser->opts.indent_step = -1;
  }
  int message_symbol_index = -1;
  for (unsigned i = 0; i < number_of_symbols; ++i) {
    if (strcmp(filenames[i], "Message.fbs") == 0) {  // NOLINT
      message_symbol_index = i;
    } else if (!parser->Parse(symbols[i], nullptr, filenames[i])) {  // NOLINT
      printf("error: %s\n", parser->error_.c_str());
      throw std::runtime_error("flatbuffer protocol definitions parser error");
    }
  }
  if (message_symbol_index == -1 ||
      !parser->Parse(symbols[message_symbol_index])) {  // NOLINT
    printf("error: %s\n", parser->error_.c_str());
    throw std::runtime_error("flatbuffer protocol definitions parser error");
  }
  return parser;
}

reflection::Schema const& init_schema(Parser& parser) {
  parser.Serialize();
  return *reflection::GetSchema(parser.builder_.GetBufferPointer());
}

static std::unique_ptr<Parser> json_parser = init_parser();
static std::unique_ptr<Parser> compact_json_parser = init_parser(true);
static std::unique_ptr<Parser> reflection_parser = init_parser();
static reflection::Schema const& schema = init_schema(*reflection_parser);

std::string message::to_json(bool compact) const {
  std::string json;
  flatbuffers::GenerateText(compact ? *compact_json_parser : *json_parser,
                            data(), &json);
  return json;
}

reflection::Schema const& message::get_schema() { return schema; }

reflection::Object const* message::get_objectref(char const* name) {
  return get_schema().objects()->LookupByKey(name);
}

std::pair<const char**, size_t> message::get_fbs_definitions() {
  return std::make_pair(symbols, number_of_symbols);
}

msg_ptr make_msg(std::string const& json, bool const fix,
                 std::size_t const fbs_max_depth,
                 std::size_t const fbs_max_tables) {
  if (json.empty()) {
    LOG(motis::logging::error) << "empty request";
    throw std::system_error(error::unable_to_parse_msg);
  }

  bool parse_ok =
      json_parser->Parse(fix ? fix_json(json).c_str() : json.c_str());
  if (!parse_ok) {
    LOG(motis::logging::error) << "parse error: " << json_parser->error_;
    throw std::system_error(error::unable_to_parse_msg);
  }

  flatbuffers::Verifier verifier(json_parser->builder_.GetBufferPointer(),
                                 json_parser->builder_.GetSize(), fbs_max_depth,
                                 fbs_max_tables);
  if (!VerifyMessageBuffer(verifier)) {
    throw std::system_error(error::malformed_msg);
  }
  auto size = json_parser->builder_.GetSize();
  auto buffer = json_parser->builder_.ReleaseBufferPointer();

  json_parser->builder_.Clear();
  return std::make_shared<message>(size, std::move(buffer));
}

msg_ptr make_msg(message_creator& builder) {
  auto len = builder.GetSize();
  auto mem = builder.ReleaseBufferPointer();
  builder.Clear();
  return std::make_shared<message>(len, std::move(mem));
}

msg_ptr make_msg(void const* buf, size_t len, std::size_t const fbs_max_depth,
                 std::size_t const fbs_max_tables) {
  auto msg = std::make_shared<message>(len, buf);

  flatbuffers::Verifier verifier(msg->data(), msg->size(), fbs_max_depth,
                                 fbs_max_tables);
  if (!VerifyMessageBuffer(verifier)) {
    throw std::system_error(error::malformed_msg);
  }

  return msg;
}

msg_ptr make_no_msg(std::string const& target, int id) {
  message_creator b;
  b.create_and_finish(MsgContent_MotisNoMessage,
                      CreateMotisNoMessage(b).Union(), target,
                      DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_success_msg(std::string const& target, int id) {
  message_creator b;
  b.create_and_finish(MsgContent_MotisSuccess, CreateMotisSuccess(b).Union(),
                      target, DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_error_msg(std::error_code const& ec, int id) {
  message_creator b;
  b.create_and_finish(
      MsgContent_MotisError,
      CreateMotisError(b, ec.value(), b.CreateString(ec.category().name()),
                       b.CreateString(ec.message()))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_unknown_error_msg(std::string const& reason, int id) {
  message_creator b;
  b.create_and_finish(
      MsgContent_MotisError,
      CreateMotisError(b, std::numeric_limits<uint16_t>::max(),
                       b.CreateString("unknown"), b.CreateString(reason))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(b);
}

}  // namespace motis::module
