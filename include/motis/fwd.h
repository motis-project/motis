#pragma once

namespace adr {
struct reverse;
struct area_database;
struct typeahead;
struct cache;
}  // namespace adr

namespace osr {

struct ways;
struct platforms;
struct lookup;
struct elevation_storage;

}  // namespace osr

namespace nigiri {

struct timetable;
struct rt_timetable;
struct shapes_storage;

namespace rt {
struct run;
struct run_stop;
}  // namespace rt

namespace routing {
struct td_offset;
struct offset;
}  // namespace routing

}  // namespace nigiri

namespace motis {

struct tiles_data;
struct rt;
struct tag_lookup;
struct config;
struct railviz_static_index;
struct railviz_rt_index;
struct elevators;
struct data;

namespace odm {
struct bounds;
}

namespace gbfs {
struct gbfs_data;
struct gbfs_provider;
struct gbfs_products_ref;
struct gbfs_routing_data;
struct provider_routing_data;
struct products_routing_data;
}  // namespace gbfs

}  // namespace motis
