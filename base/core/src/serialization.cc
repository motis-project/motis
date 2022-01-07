#include "motis/core/schedule/serialization.h"

#include "cista/serialization.h"

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/common/logging.h"

namespace motis {

static_assert(cista::to_tuple_works_v<bitfield>);

constexpr auto const MODE = cista::mode::WITH_INTEGRITY;

template <typename Ctx>
inline void serialize(Ctx& c, motis::time const* origin,
                      cista::offset_t const offset) {}

template <typename Ctx>
inline void deserialize(Ctx const& c, motis::time* el) {
  deserialize(c, &el->day_idx_);
  deserialize(c, &el->mam_);
}

cista::hash_t type_hash(motis::time const& el, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.day_idx_, h, done),
                             cista::type_hash(el.mam_, h, done));
}

template <typename Ctx>
inline void serialize(Ctx& c, bitfield_idx_or_ptr const* origin,
                      cista::offset_t const offset) {
  serialize(c, &origin->bitfield_idx_,
            offset + offsetof(bitfield_idx_or_ptr, bitfield_idx_));
}

template <typename Ctx>
inline void deserialize(Ctx const& c, bitfield_idx_or_ptr* el) {
  deserialize(c, &el->bitfield_idx_);
}

cista::hash_t type_hash(bitfield_idx_or_ptr const& el, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.bitfield_idx_, h, done));
}

template <typename Ctx>
inline void serialize(Ctx& c, static_light_connection const* origin,
                      cista::offset_t const offset) {
  serialize(c, &origin->full_con_,
            offset + offsetof(static_light_connection, full_con_));
}

template <typename Ctx>
inline void deserialize(Ctx const& c, static_light_connection* el) {
  deserialize(c, &el->full_con_);
}

cista::hash_t type_hash(static_light_connection const& el,
                        cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.full_con_, h, done),
                             cista::type_hash(el.d_time_, h, done),
                             cista::type_hash(el.a_time_, h, done),
                             cista::type_hash(el.trips_, h, done));
}

template <typename Ctx>
inline void serialize(Ctx& c, rt_light_connection const* origin,
                      cista::offset_t const offset) {
  serialize(c, &origin->full_con_,
            offset + offsetof(rt_light_connection, full_con_));
}

template <typename Ctx>
inline void deserialize(Ctx const& c, rt_light_connection* el) {
  deserialize(c, &el->full_con_);
}

cista::hash_t type_hash(rt_light_connection const& el, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.full_con_, h, done),
                             cista::type_hash(el.d_time_, h, done),
                             cista::type_hash(el.a_time_, h, done),
                             cista::type_hash(el.trips_, h, done), 31,
                             cista::type_hash(el.valid_, h, done), 1);
}

template <typename Ctx>
inline void serialize(Ctx&, primary_trip_id const*, cista::offset_t const) {}

template <typename Ctx>
inline void deserialize(Ctx const&, primary_trip_id*) {}

cista::hash_t type_hash(primary_trip_id const&, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(uint64_t{}, h, done), 31, 16, 17);
}

template <typename Ctx>
inline void serialize(Ctx& c, trip_info::route_edge const* origin,
                      cista::offset_t const offset) {
  cista::serialize(c, &origin->route_node_,
                   offset + offsetof(trip_info::route_edge, route_node_));
}

template <typename Ctx>
inline void deserialize(Ctx const& c, trip_info::route_edge* el) {
  cista::deserialize(c, &el->route_node_);
}

cista::hash_t type_hash(trip_info::route_edge const& el, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.route_node_, h, done),
                             cista::type_hash(el.outgoing_edge_idx_, h, done));
}

cista::hash_t type_hash(bitfield const&, cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>&) {
  return cista::hash_combine(cista::hash("bitfield"), h);
}

template <typename Ctx, typename T, typename SizeType>
inline void serialize(Ctx& c, dynamic_fws_multimap<T, SizeType> const* origin,
                      cista::offset_t const offset) {
  using Type = dynamic_fws_multimap<T, SizeType>;
  cista::serialize(c, &origin->index_, offset + offsetof(Type, index_));
  cista::serialize(c, &origin->data_, offset + offsetof(Type, data_));
  cista::serialize(c, &origin->element_count_,
                   offset + offsetof(Type, element_count_));
  cista::serialize(c, &origin->initial_capacity_,
                   offset + offsetof(Type, initial_capacity_));
  cista::serialize(c, &origin->growth_factor_,
                   offset + offsetof(Type, growth_factor_));
}

template <typename Ctx, typename T, typename SizeType>
inline void deserialize(Ctx const& c, dynamic_fws_multimap<T, SizeType>* el) {
  cista::deserialize(c, &el->index_);
  cista::deserialize(c, &el->data_);
  cista::deserialize(c, &el->element_count_);
  cista::deserialize(c, &el->initial_capacity_);
  cista::deserialize(c, &el->growth_factor_);
}

template <typename T, typename SizeType>
cista::hash_t type_hash(dynamic_fws_multimap<T, SizeType> const& el,
                        cista::hash_t const h,
                        std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.index_, h, done),
                             cista::type_hash(el.data_, h, done),
                             cista::type_hash(el.element_count_, h, done),
                             cista::type_hash(el.initial_capacity_, h, done),
                             cista::type_hash(el.growth_factor_, h, done));
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

schedule_ptr read_graph(std::string const& path, cista::memory_holder& mem,
                        bool const read_mmap) {
  if (read_mmap) {
    auto mmap = cista::mmap{path.c_str(), cista::mmap::protection::READ};
    mem = cista::buf<cista::mmap>(std::move(mmap));
  } else {
    mem = cista::file(path.c_str(), "r").content();
  }

  auto ptr = schedule_ptr{};
  ptr.self_allocated_ = false;
  ptr.el_ =
      std::visit(overloaded{[&](cista::buf<cista::mmap>& b) {
                              return reinterpret_cast<schedule*>(
                                  &b[cista::data_start(MODE)]);
                            },
                            [&](cista::buffer& b) {
                              return cista::deserialize<schedule, MODE>(b);
                            },
                            [&](cista::byte_buf& b) {
                              return cista::deserialize<schedule, MODE>(b);
                            }},
                 mem);
  return ptr;
}

void write_graph(std::string const& path, schedule const& sched) {
  auto mmap = cista::mmap{path.c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));

  {
    logging::scoped_timer t{"writing graph"};
    cista::serialize<MODE>(writer, sched);
  }
}

}  // namespace motis
