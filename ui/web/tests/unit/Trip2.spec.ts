/* eslint-disable @typescript-eslint/no-explicit-any */
import TripResponseContent from "@/models/TripResponseContent";
import store from "@/store";
import { config, VueWrapper, mount } from "@vue/test-utils";
import flushPromises from "flush-promises";
import Trip from "../../src/views/Trip.vue"

describe('Test Trip.vue with multiple connections', () => {
  let wrapper: VueWrapper;

  beforeAll(() => {
    store.state.connections = mockConnections;
    store.state.startInput = {
      name: "karo 5",
      pos: {
        lat: 0,
        lng: 0
      },
      type: "",
      regions: []
    };
    store.state.destinationInput = {
      name: "Marburger Schloß",
      pos: {
        lat: 0,
        lng: 0
      },
      type: "",
      regions: []
    };
    config.global.mocks = {
      $route: {
        name: "Connection"
      }
    };
  })

  beforeEach(async () => {
    wrapper = mount(Trip, {
      props: {
        index: 2,
      },
      global: {
        plugins: [store]
      }
    });
    await flushPromises();
  })

  it("Snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Loads content", () => {
    expect((wrapper.vm.$data as any).content).toStrictEqual(mockConnections[2]);
    expect((wrapper.vm.$data as any).isContentLoaded).toBe(true);
  })

  it("Expand button works", async () => {
    const button = wrapper.findAll(".intermediate-stops-toggle")[1];
    const list = wrapper.findAll(".intermediate-stops")[0];
    expect(list.classes()).not.toContain("expanded");
    button.trigger("click");
    await flushPromises();
    expect(list.classes()).toContain("expanded");
    button.trigger("click");
    await flushPromises();
    expect(list.classes()).not.toContain("expanded");
  })
});


const mockConnections: TripResponseContent[] = [
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644332820,
          "schedule_time": 1644332820,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:24747:3:3",
          "name": "Darmstadt Willy-Brandt-Platz",
          "pos": {
            "lat": 49.876114,
            "lng": 8.650171
          }
        },
        "arrival": {
          "time": 1644333300,
          "schedule_time": 1644333300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333420,
          "schedule_time": 1644333420,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:16019:1:1",
          "name": "Darmstadt Kasinostraße",
          "pos": {
            "lat": 49.875141,
            "lng": 8.643821
          }
        },
        "arrival": {
          "time": 1644333480,
          "schedule_time": 1644333480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333480,
          "schedule_time": 1644333480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16014:1:1",
          "name": "Darmstadt Kirschenallee",
          "pos": {
            "lat": 49.874908,
            "lng": 8.635002
          }
        },
        "arrival": {
          "time": 1644333540,
          "schedule_time": 1644333540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333540,
          "schedule_time": 1644333540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:49:49",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872295,
            "lng": 8.631474
          }
        },
        "arrival": {
          "time": 1644333720,
          "schedule_time": 1644333720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333720,
          "schedule_time": 1644333720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:41:65",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644333840,
          "schedule_time": 1644333840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333840,
          "schedule_time": 1644333840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06412:10:17:18",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106468,
            "lng": 8.662687
          }
        },
        "arrival": {
          "time": 1644334800,
          "schedule_time": 1644334800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335280,
          "schedule_time": 1644335280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:14:14",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.119385,
            "lng": 8.638952
          }
        },
        "arrival": {
          "time": 1644335580,
          "schedule_time": 1644335580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335640,
          "schedule_time": 1644335640,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644336900,
          "schedule_time": 1644336900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337020,
          "schedule_time": 1644337020,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644337980,
          "schedule_time": 1644337980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338040,
          "schedule_time": 1644338040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:10011:6:11",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.818718,
            "lng": 8.774009
          }
        },
        "arrival": {
          "time": 1644338940,
          "schedule_time": 1644338940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338940,
          "schedule_time": 1644338940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321001131",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.819252,
            "lng": 8.774304
          }
        },
        "arrival": {
          "time": 1644339060,
          "schedule_time": 1644339060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339060,
          "schedule_time": 1644339060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.819545,
            "lng": 8.774208
          }
        },
        "arrival": {
          "time": 1644339120,
          "schedule_time": 1644339120,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339120,
          "schedule_time": 1644339120,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644339540,
          "schedule_time": 1644339540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 4
          },
          "category_name": "Bus",
          "category_id": 0,
          "clasz": 10,
          "train_nr": 0,
          "line_id": "GB",
          "name": "Bus GB",
          "provider": "FS Omnibus GmbH & Co. KG",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 4,
            "to": 5
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 5,
            "to": 10
          },
          "category_name": "IC",
          "category_id": 10,
          "clasz": 2,
          "train_nr": 2184,
          "line_id": "26",
          "name": "IC 2184",
          "provider": "DB Fernverkehr AG",
          "direction": "Hamburg-Altona"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 10,
            "to": 11
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 11,
            "to": 12
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 12,
            "to": 13
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 4
        },
        "id": {
          "station_id": "delfi_de:06432:16120:1:1",
          "train_nr": 93623,
          "time": 1644330900,
          "target_station_id": "delfi_de:06411:4734:49:49",
          "target_time": 1644333720,
          "line_id": "GB"
        },
      },
      {
        "range": {
          "from": 5,
          "to": 10
        },
        "id": {
          "station_id": "delfi_de:08212:90",
          "train_nr": 2184,
          "time": 1644329400,
          "target_station_id": "delfi_de:02000:8002553",
          "target_time": 1644353160,
          "line_id": "26"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333060,
          "schedule_time": 1644333060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:24747:3:3",
          "name": "Darmstadt Willy-Brandt-Platz",
          "pos": {
            "lat": 49.876114,
            "lng": 8.650171
          }
        },
        "arrival": {
          "time": 1644333540,
          "schedule_time": 1644333540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333660,
          "schedule_time": 1644333660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:24919:3:3",
          "name": "Darmstadt Klinikum",
          "pos": {
            "lat": 49.875797,
            "lng": 8.648241
          }
        },
        "arrival": {
          "time": 1644333720,
          "schedule_time": 1644333720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333720,
          "schedule_time": 1644333720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16019:1:1",
          "name": "Darmstadt Kasinostraße",
          "pos": {
            "lat": 49.875141,
            "lng": 8.643821
          }
        },
        "arrival": {
          "time": 1644333780,
          "schedule_time": 1644333780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333780,
          "schedule_time": 1644333780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16014:1:1",
          "name": "Darmstadt Kirschenallee",
          "pos": {
            "lat": 49.874908,
            "lng": 8.635002
          }
        },
        "arrival": {
          "time": 1644333840,
          "schedule_time": 1644333840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333840,
          "schedule_time": 1644333840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644333960,
          "schedule_time": 1644333960,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333960,
          "schedule_time": 1644333960,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:44:66",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644334080,
          "schedule_time": 1644334080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334200,
          "schedule_time": 1644334200,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06438:2705:1:1",
          "name": "Langen (Hessen) Bahnhof",
          "pos": {
            "lat": 49.993542,
            "lng": 8.656978
          }
        },
        "arrival": {
          "time": 1644334680,
          "schedule_time": 1644334680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334740,
          "schedule_time": 1644334740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:16:12",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106297,
            "lng": 8.662814
          }
        },
        "arrival": {
          "time": 1644335280,
          "schedule_time": 1644335280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335280,
          "schedule_time": 1644335280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:18:22",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106621,
            "lng": 8.662561
          }
        },
        "arrival": {
          "time": 1644335400,
          "schedule_time": 1644335400,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336060,
          "schedule_time": 1644336060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:14:14",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.119385,
            "lng": 8.638952
          }
        },
        "arrival": {
          "time": 1644336300,
          "schedule_time": 1644336300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336300,
          "schedule_time": 1644336300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644337500,
          "schedule_time": 1644337500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337560,
          "schedule_time": 1644337560,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:11135:11:11",
          "name": "Bad Nauheim Bahnhof",
          "pos": {
            "lat": 50.367687,
            "lng": 8.749236
          }
        },
        "arrival": {
          "time": 1644337740,
          "schedule_time": 1644337740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337800,
          "schedule_time": 1644337800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:11133:10:10",
          "name": "Butzbach Bahnhof",
          "pos": {
            "lat": 50.429989,
            "lng": 8.669856
          }
        },
        "arrival": {
          "time": 1644338220,
          "schedule_time": 1644338220,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338220,
          "schedule_time": 1644338220,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644338880,
          "schedule_time": 1644338880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338940,
          "schedule_time": 1644338940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11085:2:2",
          "name": "Lollar Bahnhof",
          "pos": {
            "lat": 50.647739,
            "lng": 8.701562
          }
        },
        "arrival": {
          "time": 1644339240,
          "schedule_time": 1644339240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339300,
          "schedule_time": 1644339300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:11084:2:2",
          "name": "Fronhausen Bahnhof",
          "pos": {
            "lat": 50.706112,
            "lng": 8.699139
          }
        },
        "arrival": {
          "time": 1644339600,
          "schedule_time": 1644339600,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339600,
          "schedule_time": 1644339600,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321108436",
          "name": "Fronhausen Bahnhof",
          "pos": {
            "lat": 50.706467,
            "lng": 8.699322
          }
        },
        "arrival": {
          "time": 1644339720,
          "schedule_time": 1644339720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339720,
          "schedule_time": 1644339720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.706462,
            "lng": 8.699455
          }
        },
        "arrival": {
          "time": 1644339780,
          "schedule_time": 1644339780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339780,
          "schedule_time": 1644339780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644340620,
          "schedule_time": 1644340620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 5
          },
          "category_name": "Str",
          "category_id": 1,
          "clasz": 9,
          "train_nr": 224,
          "line_id": "3",
          "name": "3",
          "provider": "HEAG Mobilo",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 5,
            "to": 6
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 6,
            "to": 8
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 15322,
          "line_id": "RB68",
          "name": "RB68",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Frankfurt (Main) Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 8,
            "to": 9
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 9,
            "to": 16
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 4176,
          "line_id": "RE30",
          "name": "RE30",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Marburg Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 16,
            "to": 17
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 17,
            "to": 18
          },
          "mumo_id": 5431,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 18,
            "to": 19
          },
          "mumo_id": 5431,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 5
        },
        "id": {
          "station_id": "delfi_de:06411:15838:2:2",
          "train_nr": 224,
          "time": 1644332700,
          "target_station_id": "delfi_de:06411:4734:48:48",
          "target_time": 1644333960,
          "line_id": "3"
        },
      },
      {
        "range": {
          "from": 6,
          "to": 8
        },
        "id": {
          "station_id": "delfi_de:08226:4252:3:3",
          "train_nr": 15322,
          "time": 1644329040,
          "target_station_id": "delfi_de:06412:10:16:12",
          "target_time": 1644335280,
          "line_id": "RB68"
        },
      },
      {
        "range": {
          "from": 9,
          "to": 16
        },
        "id": {
          "station_id": "delfi_de:06412:10:18:22",
          "train_nr": 4176,
          "time": 1644336060,
          "target_station_id": "delfi_de:06534:10011:4:9",
          "target_time": 1644340500,
          "line_id": "RE30"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333300,
          "schedule_time": 1644333300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4736:5:5",
          "name": "Darmstadt Luisenplatz",
          "pos": {
            "lat": 49.872913,
            "lng": 8.650806
          }
        },
        "arrival": {
          "time": 1644333780,
          "schedule_time": 1644333780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644333900,
          "schedule_time": 1644333900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:24566:1:1",
          "name": "Darmstadt Rhein-/Neckarstraße",
          "pos": {
            "lat": 49.871902,
            "lng": 8.643649
          }
        },
        "arrival": {
          "time": 1644334020,
          "schedule_time": 1644334020,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334020,
          "schedule_time": 1644334020,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4735:3:3",
          "name": "Darmstadt Berliner Allee",
          "pos": {
            "lat": 49.870773,
            "lng": 8.635729
          }
        },
        "arrival": {
          "time": 1644334080,
          "schedule_time": 1644334080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334080,
          "schedule_time": 1644334080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644334260,
          "schedule_time": 1644334260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334260,
          "schedule_time": 1644334260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644334380,
          "schedule_time": 1644334380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334380,
          "schedule_time": 1644334380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:64:63",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644334500,
          "schedule_time": 1644334500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334500,
          "schedule_time": 1644334500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:4733:1:1",
          "name": "Darmstadt-Arheilgen Bahnhof",
          "pos": {
            "lat": 49.913868,
            "lng": 8.645456
          }
        },
        "arrival": {
          "time": 1644334740,
          "schedule_time": 1644334740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334740,
          "schedule_time": 1644334740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4732:3:3",
          "name": "Darmstadt-Wixhausen Bahnhof",
          "pos": {
            "lat": 49.93034,
            "lng": 8.647816
          }
        },
        "arrival": {
          "time": 1644334860,
          "schedule_time": 1644334860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644334860,
          "schedule_time": 1644334860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06432:4792:2:2",
          "name": "Erzhausen Bahnhof",
          "pos": {
            "lat": 49.951153,
            "lng": 8.650842
          }
        },
        "arrival": {
          "time": 1644334980,
          "schedule_time": 1644334980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335040,
          "schedule_time": 1644335040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2706:1:1",
          "name": "Egelsbach Bahnhof",
          "pos": {
            "lat": 49.96817,
            "lng": 8.653729
          }
        },
        "arrival": {
          "time": 1644335160,
          "schedule_time": 1644335160,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335160,
          "schedule_time": 1644335160,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2705:2:2",
          "name": "Langen (Hessen) Bahnhof",
          "pos": {
            "lat": 49.993542,
            "lng": 8.656922
          }
        },
        "arrival": {
          "time": 1644335280,
          "schedule_time": 1644335280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335340,
          "schedule_time": 1644335340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:11505:1:1",
          "name": "Langen (Hessen) Flugsicherung",
          "pos": {
            "lat": 50.005276,
            "lng": 8.658583
          }
        },
        "arrival": {
          "time": 1644335400,
          "schedule_time": 1644335400,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335460,
          "schedule_time": 1644335460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2702:2:2",
          "name": "Dreieich-Buchschlag Bahnhof",
          "pos": {
            "lat": 50.022583,
            "lng": 8.661405
          }
        },
        "arrival": {
          "time": 1644335580,
          "schedule_time": 1644335580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335580,
          "schedule_time": 1644335580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2701:2:2",
          "name": "Neu-Isenburg Bahnhof",
          "pos": {
            "lat": 50.052856,
            "lng": 8.665492
          }
        },
        "arrival": {
          "time": 1644335760,
          "schedule_time": 1644335760,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335760,
          "schedule_time": 1644335760,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:2799:11:1",
          "name": "Frankfurt (Main) Louisa Bahnhof",
          "pos": {
            "lat": 50.083672,
            "lng": 8.670308
          }
        },
        "arrival": {
          "time": 1644335940,
          "schedule_time": 1644335940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644335940,
          "schedule_time": 1644335940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:933:1:2",
          "name": "Frankfurt (Main) Stresemannallee Bahnhof",
          "pos": {
            "lat": 50.09454,
            "lng": 8.671268
          }
        },
        "arrival": {
          "time": 1644336060,
          "schedule_time": 1644336060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336060,
          "schedule_time": 1644336060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:912:2:4",
          "name": "Frankfurt (Main) Südbahnhof",
          "pos": {
            "lat": 50.099026,
            "lng": 8.685507
          }
        },
        "arrival": {
          "time": 1644336120,
          "schedule_time": 1644336120,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336180,
          "schedule_time": 1644336180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:907:11:12",
          "name": "Frankfurt (Main) Lokalbahnhof",
          "pos": {
            "lat": 50.101933,
            "lng": 8.692938
          }
        },
        "arrival": {
          "time": 1644336240,
          "schedule_time": 1644336240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336300,
          "schedule_time": 1644336300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:525:1:2",
          "name": "Frankfurt (Main) Ostendstraße",
          "pos": {
            "lat": 50.112385,
            "lng": 8.696968
          }
        },
        "arrival": {
          "time": 1644336360,
          "schedule_time": 1644336360,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336420,
          "schedule_time": 1644336420,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:510:1:3",
          "name": "Frankfurt (Main) Konstablerwache",
          "pos": {
            "lat": 50.114697,
            "lng": 8.685797
          }
        },
        "arrival": {
          "time": 1644336480,
          "schedule_time": 1644336480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336540,
          "schedule_time": 1644336540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1:11:2",
          "name": "Frankfurt (Main) Hauptwache",
          "pos": {
            "lat": 50.114056,
            "lng": 8.678139
          }
        },
        "arrival": {
          "time": 1644336600,
          "schedule_time": 1644336600,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336600,
          "schedule_time": 1644336600,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:11:1:2",
          "name": "Frankfurt (Main) Taunusanlage",
          "pos": {
            "lat": 50.113476,
            "lng": 8.668763
          }
        },
        "arrival": {
          "time": 1644336660,
          "schedule_time": 1644336660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336720,
          "schedule_time": 1644336720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:7010:2:1",
          "name": "Frankfurt (Main) Hauptbahnhof tief",
          "pos": {
            "lat": 50.107132,
            "lng": 8.662473
          }
        },
        "arrival": {
          "time": 1644336780,
          "schedule_time": 1644336780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336780,
          "schedule_time": 1644336780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:19:24",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106758,
            "lng": 8.662378
          }
        },
        "arrival": {
          "time": 1644336900,
          "schedule_time": 1644336900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337140,
          "schedule_time": 1644337140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644338640,
          "schedule_time": 1644338640,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338700,
          "schedule_time": 1644338700,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644339780,
          "schedule_time": 1644339780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339840,
          "schedule_time": 1644339840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:10011:6:11",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.818718,
            "lng": 8.774009
          }
        },
        "arrival": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321001131",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.819252,
            "lng": 8.774304
          }
        },
        "arrival": {
          "time": 1644340860,
          "schedule_time": 1644340860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340860,
          "schedule_time": 1644340860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.819545,
            "lng": 8.774208
          }
        },
        "arrival": {
          "time": 1644340920,
          "schedule_time": 1644340920,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340920,
          "schedule_time": 1644340920,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644341340,
          "schedule_time": 1644341340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 5
          },
          "category_name": "Str",
          "category_id": 1,
          "clasz": 9,
          "train_nr": 310,
          "line_id": "5",
          "name": "5",
          "provider": "HEAG Mobilo",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 5,
            "to": 6
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 6,
            "to": 23
          },
          "category_name": "S",
          "category_id": 7,
          "clasz": 7,
          "train_nr": 35354,
          "line_id": "S3",
          "name": "S3",
          "provider": "DB Regio AG S-Bahn Rhein-Main",
          "direction": "Bad Soden(Taunus)"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 23,
            "to": 24
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 24,
            "to": 27
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 4162,
          "line_id": "RE30",
          "name": "RE30",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Kassel Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 27,
            "to": 28
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 28,
            "to": 29
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 29,
            "to": 30
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 5
        },
        "id": {
          "station_id": "delfi_de:06411:11328:3:3",
          "train_nr": 310,
          "time": 1644332880,
          "target_station_id": "delfi_de:06411:4734:48:48",
          "target_time": 1644334380,
          "line_id": "5"
        },
      },
      {
        "range": {
          "from": 6,
          "to": 23
        },
        "id": {
          "station_id": "delfi_de:06411:4734:64:63",
          "train_nr": 35354,
          "time": 1644334500,
          "target_station_id": "delfi_de:06436:4299:1:2",
          "target_time": 1644338340,
          "line_id": "S3"
        },
      },
      {
        "range": {
          "from": 24,
          "to": 27
        },
        "id": {
          "station_id": "delfi_de:06412:10:19:24",
          "train_nr": 4162,
          "time": 1644337140,
          "target_station_id": "delfi_de:06611:200001",
          "target_time": 1644345240,
          "line_id": "RE30"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336660,
          "schedule_time": 1644336660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:24747:3:3",
          "name": "Darmstadt Willy-Brandt-Platz",
          "pos": {
            "lat": 49.876114,
            "lng": 8.650171
          }
        },
        "arrival": {
          "time": 1644337140,
          "schedule_time": 1644337140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337260,
          "schedule_time": 1644337260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:24919:3:3",
          "name": "Darmstadt Klinikum",
          "pos": {
            "lat": 49.875797,
            "lng": 8.648241
          }
        },
        "arrival": {
          "time": 1644337320,
          "schedule_time": 1644337320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337320,
          "schedule_time": 1644337320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16019:1:1",
          "name": "Darmstadt Kasinostraße",
          "pos": {
            "lat": 49.875141,
            "lng": 8.643821
          }
        },
        "arrival": {
          "time": 1644337380,
          "schedule_time": 1644337380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337380,
          "schedule_time": 1644337380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16014:1:1",
          "name": "Darmstadt Kirschenallee",
          "pos": {
            "lat": 49.874908,
            "lng": 8.635002
          }
        },
        "arrival": {
          "time": 1644337440,
          "schedule_time": 1644337440,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337440,
          "schedule_time": 1644337440,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644337560,
          "schedule_time": 1644337560,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337560,
          "schedule_time": 1644337560,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:44:66",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644337680,
          "schedule_time": 1644337680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337800,
          "schedule_time": 1644337800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06438:2705:1:1",
          "name": "Langen (Hessen) Bahnhof",
          "pos": {
            "lat": 49.993542,
            "lng": 8.656978
          }
        },
        "arrival": {
          "time": 1644338280,
          "schedule_time": 1644338280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338340,
          "schedule_time": 1644338340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:16:12",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106297,
            "lng": 8.662814
          }
        },
        "arrival": {
          "time": 1644338880,
          "schedule_time": 1644338880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338880,
          "schedule_time": 1644338880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:18:22",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106621,
            "lng": 8.662561
          }
        },
        "arrival": {
          "time": 1644339000,
          "schedule_time": 1644339000,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339060,
          "schedule_time": 1644339060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:14:14",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.119385,
            "lng": 8.638952
          }
        },
        "arrival": {
          "time": 1644339240,
          "schedule_time": 1644339240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339300,
          "schedule_time": 1644339300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644340440,
          "schedule_time": 1644340440,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340500,
          "schedule_time": 1644340500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:11135:11:11",
          "name": "Bad Nauheim Bahnhof",
          "pos": {
            "lat": 50.367687,
            "lng": 8.749236
          }
        },
        "arrival": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340800,
          "schedule_time": 1644340800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:11133:10:10",
          "name": "Butzbach Bahnhof",
          "pos": {
            "lat": 50.429989,
            "lng": 8.669856
          }
        },
        "arrival": {
          "time": 1644341100,
          "schedule_time": 1644341100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341160,
          "schedule_time": 1644341160,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644341700,
          "schedule_time": 1644341700,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341940,
          "schedule_time": 1644341940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:14672:2:2",
          "name": "Gießen Oswaldsgarten",
          "pos": {
            "lat": 50.586987,
            "lng": 8.669891
          }
        },
        "arrival": {
          "time": 1644342000,
          "schedule_time": 1644342000,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342060,
          "schedule_time": 1644342060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11085:2:2",
          "name": "Lollar Bahnhof",
          "pos": {
            "lat": 50.647739,
            "lng": 8.701562
          }
        },
        "arrival": {
          "time": 1644342240,
          "schedule_time": 1644342240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342300,
          "schedule_time": 1644342300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11011:2:2",
          "name": "Lollar-Odenhausen Friedelhausen Bf",
          "pos": {
            "lat": 50.671925,
            "lng": 8.710985
          }
        },
        "arrival": {
          "time": 1644342420,
          "schedule_time": 1644342420,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342480,
          "schedule_time": 1644342480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:11084:2:2",
          "name": "Fronhausen Bahnhof",
          "pos": {
            "lat": 50.706112,
            "lng": 8.699139
          }
        },
        "arrival": {
          "time": 1644342660,
          "schedule_time": 1644342660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342660,
          "schedule_time": 1644342660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321108436",
          "name": "Fronhausen Bahnhof",
          "pos": {
            "lat": 50.706467,
            "lng": 8.699322
          }
        },
        "arrival": {
          "time": 1644342780,
          "schedule_time": 1644342780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342780,
          "schedule_time": 1644342780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.706462,
            "lng": 8.699455
          }
        },
        "arrival": {
          "time": 1644342840,
          "schedule_time": 1644342840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342840,
          "schedule_time": 1644342840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644343680,
          "schedule_time": 1644343680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 5
          },
          "category_name": "Str",
          "category_id": 1,
          "clasz": 9,
          "train_nr": 240,
          "line_id": "3",
          "name": "3",
          "provider": "HEAG Mobilo",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 5,
            "to": 6
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 6,
            "to": 8
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 15324,
          "line_id": "RB68",
          "name": "RB68",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Frankfurt (Main) Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 8,
            "to": 9
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 9,
            "to": 18
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 15024,
          "line_id": "RB41",
          "name": "RB41",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Treysa"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 18,
            "to": 19
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 19,
            "to": 20
          },
          "mumo_id": 5431,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 20,
            "to": 21
          },
          "mumo_id": 5431,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 5
        },
        "id": {
          "station_id": "delfi_de:06411:15838:2:2",
          "train_nr": 240,
          "time": 1644336300,
          "target_station_id": "delfi_de:06411:4734:48:48",
          "target_time": 1644337560,
          "line_id": "3"
        },
      },
      {
        "range": {
          "from": 6,
          "to": 8
        },
        "id": {
          "station_id": "delfi_de:08226:4252:3:3",
          "train_nr": 15324,
          "time": 1644332640,
          "target_station_id": "delfi_de:06412:10:16:12",
          "target_time": 1644338880,
          "line_id": "RB68"
        },
      },
      {
        "range": {
          "from": 9,
          "to": 18
        },
        "id": {
          "station_id": "delfi_de:06412:10:18:22",
          "train_nr": 15024,
          "time": 1644339060,
          "target_station_id": "delfi_de:06634:204662",
          "target_time": 1644345720,
          "line_id": "RB41"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644336900,
          "schedule_time": 1644336900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4736:5:5",
          "name": "Darmstadt Luisenplatz",
          "pos": {
            "lat": 49.872913,
            "lng": 8.650806
          }
        },
        "arrival": {
          "time": 1644337380,
          "schedule_time": 1644337380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337500,
          "schedule_time": 1644337500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:24566:1:1",
          "name": "Darmstadt Rhein-/Neckarstraße",
          "pos": {
            "lat": 49.871902,
            "lng": 8.643649
          }
        },
        "arrival": {
          "time": 1644337620,
          "schedule_time": 1644337620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337620,
          "schedule_time": 1644337620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4735:3:3",
          "name": "Darmstadt Berliner Allee",
          "pos": {
            "lat": 49.870773,
            "lng": 8.635729
          }
        },
        "arrival": {
          "time": 1644337680,
          "schedule_time": 1644337680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337680,
          "schedule_time": 1644337680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644337860,
          "schedule_time": 1644337860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337860,
          "schedule_time": 1644337860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644337980,
          "schedule_time": 1644337980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644337980,
          "schedule_time": 1644337980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:64:63",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644338100,
          "schedule_time": 1644338100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338100,
          "schedule_time": 1644338100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:4733:1:1",
          "name": "Darmstadt-Arheilgen Bahnhof",
          "pos": {
            "lat": 49.913868,
            "lng": 8.645456
          }
        },
        "arrival": {
          "time": 1644338340,
          "schedule_time": 1644338340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338340,
          "schedule_time": 1644338340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4732:3:3",
          "name": "Darmstadt-Wixhausen Bahnhof",
          "pos": {
            "lat": 49.93034,
            "lng": 8.647816
          }
        },
        "arrival": {
          "time": 1644338460,
          "schedule_time": 1644338460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338460,
          "schedule_time": 1644338460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06432:4792:2:2",
          "name": "Erzhausen Bahnhof",
          "pos": {
            "lat": 49.951153,
            "lng": 8.650842
          }
        },
        "arrival": {
          "time": 1644338580,
          "schedule_time": 1644338580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338640,
          "schedule_time": 1644338640,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2706:1:1",
          "name": "Egelsbach Bahnhof",
          "pos": {
            "lat": 49.96817,
            "lng": 8.653729
          }
        },
        "arrival": {
          "time": 1644338760,
          "schedule_time": 1644338760,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338760,
          "schedule_time": 1644338760,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2705:2:2",
          "name": "Langen (Hessen) Bahnhof",
          "pos": {
            "lat": 49.993542,
            "lng": 8.656922
          }
        },
        "arrival": {
          "time": 1644338880,
          "schedule_time": 1644338880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644338940,
          "schedule_time": 1644338940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:11505:1:1",
          "name": "Langen (Hessen) Flugsicherung",
          "pos": {
            "lat": 50.005276,
            "lng": 8.658583
          }
        },
        "arrival": {
          "time": 1644339000,
          "schedule_time": 1644339000,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339060,
          "schedule_time": 1644339060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2702:2:2",
          "name": "Dreieich-Buchschlag Bahnhof",
          "pos": {
            "lat": 50.022583,
            "lng": 8.661405
          }
        },
        "arrival": {
          "time": 1644339180,
          "schedule_time": 1644339180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339180,
          "schedule_time": 1644339180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2701:2:2",
          "name": "Neu-Isenburg Bahnhof",
          "pos": {
            "lat": 50.052856,
            "lng": 8.665492
          }
        },
        "arrival": {
          "time": 1644339360,
          "schedule_time": 1644339360,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339360,
          "schedule_time": 1644339360,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:2799:11:1",
          "name": "Frankfurt (Main) Louisa Bahnhof",
          "pos": {
            "lat": 50.083672,
            "lng": 8.670308
          }
        },
        "arrival": {
          "time": 1644339540,
          "schedule_time": 1644339540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339540,
          "schedule_time": 1644339540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:933:1:2",
          "name": "Frankfurt (Main) Stresemannallee Bahnhof",
          "pos": {
            "lat": 50.09454,
            "lng": 8.671268
          }
        },
        "arrival": {
          "time": 1644339660,
          "schedule_time": 1644339660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339660,
          "schedule_time": 1644339660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:912:2:4",
          "name": "Frankfurt (Main) Südbahnhof",
          "pos": {
            "lat": 50.099026,
            "lng": 8.685507
          }
        },
        "arrival": {
          "time": 1644339720,
          "schedule_time": 1644339720,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339780,
          "schedule_time": 1644339780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:907:11:12",
          "name": "Frankfurt (Main) Lokalbahnhof",
          "pos": {
            "lat": 50.101933,
            "lng": 8.692938
          }
        },
        "arrival": {
          "time": 1644339840,
          "schedule_time": 1644339840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644339900,
          "schedule_time": 1644339900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:525:1:2",
          "name": "Frankfurt (Main) Ostendstraße",
          "pos": {
            "lat": 50.112385,
            "lng": 8.696968
          }
        },
        "arrival": {
          "time": 1644339960,
          "schedule_time": 1644339960,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340020,
          "schedule_time": 1644340020,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:510:1:3",
          "name": "Frankfurt (Main) Konstablerwache",
          "pos": {
            "lat": 50.114697,
            "lng": 8.685797
          }
        },
        "arrival": {
          "time": 1644340080,
          "schedule_time": 1644340080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340140,
          "schedule_time": 1644340140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1:11:2",
          "name": "Frankfurt (Main) Hauptwache",
          "pos": {
            "lat": 50.114056,
            "lng": 8.678139
          }
        },
        "arrival": {
          "time": 1644340200,
          "schedule_time": 1644340200,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340200,
          "schedule_time": 1644340200,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:11:1:2",
          "name": "Frankfurt (Main) Taunusanlage",
          "pos": {
            "lat": 50.113476,
            "lng": 8.668763
          }
        },
        "arrival": {
          "time": 1644340260,
          "schedule_time": 1644340260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340320,
          "schedule_time": 1644340320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:7010:2:1",
          "name": "Frankfurt (Main) Hauptbahnhof tief",
          "pos": {
            "lat": 50.107132,
            "lng": 8.662473
          }
        },
        "arrival": {
          "time": 1644340380,
          "schedule_time": 1644340380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340440,
          "schedule_time": 1644340440,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:101:11:1",
          "name": "Frankfurt (Main) Galluswarte",
          "pos": {
            "lat": 50.104195,
            "lng": 8.644517
          }
        },
        "arrival": {
          "time": 1644340560,
          "schedule_time": 1644340560,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340620,
          "schedule_time": 1644340620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:11501:1:1",
          "name": "Frankfurt (Main) Messe",
          "pos": {
            "lat": 50.111313,
            "lng": 8.643585
          }
        },
        "arrival": {
          "time": 1644340680,
          "schedule_time": 1644340680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340680,
          "schedule_time": 1644340680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:11:11",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.11935,
            "lng": 8.639483
          }
        },
        "arrival": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:14:14",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.119385,
            "lng": 8.638952
          }
        },
        "arrival": {
          "time": 1644340860,
          "schedule_time": 1644340860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341040,
          "schedule_time": 1644341040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644342240,
          "schedule_time": 1644342240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342300,
          "schedule_time": 1644342300,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:11:11",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644343320,
          "schedule_time": 1644343320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343500,
          "schedule_time": 1644343500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:10011:6:11",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.818718,
            "lng": 8.774009
          }
        },
        "arrival": {
          "time": 1644344340,
          "schedule_time": 1644344340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644344340,
          "schedule_time": 1644344340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321001131",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.819252,
            "lng": 8.774304
          }
        },
        "arrival": {
          "time": 1644344460,
          "schedule_time": 1644344460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644344460,
          "schedule_time": 1644344460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.819545,
            "lng": 8.774208
          }
        },
        "arrival": {
          "time": 1644344520,
          "schedule_time": 1644344520,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644344520,
          "schedule_time": 1644344520,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644344940,
          "schedule_time": 1644344940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 5
          },
          "category_name": "Str",
          "category_id": 1,
          "clasz": 9,
          "train_nr": 334,
          "line_id": "5",
          "name": "5",
          "provider": "HEAG Mobilo",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 5,
            "to": 6
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 6,
            "to": 26
          },
          "category_name": "S",
          "category_id": 7,
          "clasz": 7,
          "train_nr": 35358,
          "line_id": "S3",
          "name": "S3",
          "provider": "DB Regio AG S-Bahn Rhein-Main",
          "direction": "Bad Soden(Taunus)"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 26,
            "to": 27
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 27,
            "to": 30
          },
          "category_name": "DPN",
          "category_id": 5,
          "clasz": 6,
          "train_nr": 24418,
          "line_id": "RE98",
          "name": "RE98",
          "provider": "Hessische Landesbahn GmbH",
          "direction": "Kassel Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 30,
            "to": 31
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 31,
            "to": 32
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 32,
            "to": 33
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 5
        },
        "id": {
          "station_id": "delfi_de:06411:11328:3:3",
          "train_nr": 334,
          "time": 1644336480,
          "target_station_id": "delfi_de:06411:4734:48:48",
          "target_time": 1644337980,
          "line_id": "5"
        },
      },
      {
        "range": {
          "from": 6,
          "to": 26
        },
        "id": {
          "station_id": "delfi_de:06411:4734:64:63",
          "train_nr": 35358,
          "time": 1644338100,
          "target_station_id": "delfi_de:06436:4299:1:2",
          "target_time": 1644341940,
          "line_id": "S3"
        },
      },
      {
        "range": {
          "from": 27,
          "to": 30
        },
        "id": {
          "station_id": "delfi_de:06412:10:19:24",
          "train_nr": 24418,
          "time": 1644340800,
          "target_station_id": "delfi_de:06611:200001",
          "target_time": 1644350040,
          "line_id": "RE98"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340020,
          "schedule_time": 1644340020,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:24747:3:3",
          "name": "Darmstadt Willy-Brandt-Platz",
          "pos": {
            "lat": 49.876114,
            "lng": 8.650171
          }
        },
        "arrival": {
          "time": 1644340500,
          "schedule_time": 1644340500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340620,
          "schedule_time": 1644340620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:16019:1:1",
          "name": "Darmstadt Kasinostraße",
          "pos": {
            "lat": 49.875141,
            "lng": 8.643821
          }
        },
        "arrival": {
          "time": 1644340680,
          "schedule_time": 1644340680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340680,
          "schedule_time": 1644340680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:16014:1:1",
          "name": "Darmstadt Kirschenallee",
          "pos": {
            "lat": 49.874908,
            "lng": 8.635002
          }
        },
        "arrival": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340740,
          "schedule_time": 1644340740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:49:49",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872295,
            "lng": 8.631474
          }
        },
        "arrival": {
          "time": 1644340920,
          "schedule_time": 1644340920,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340920,
          "schedule_time": 1644340920,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:41:65",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644341040,
          "schedule_time": 1644341040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341040,
          "schedule_time": 1644341040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06412:10:16:16",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106365,
            "lng": 8.662772
          }
        },
        "arrival": {
          "time": 1644342000,
          "schedule_time": 1644342000,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342480,
          "schedule_time": 1644342480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1204:14:14",
          "name": "Frankfurt (Main) Westbahnhof",
          "pos": {
            "lat": 50.119385,
            "lng": 8.638952
          }
        },
        "arrival": {
          "time": 1644342780,
          "schedule_time": 1644342780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342840,
          "schedule_time": 1644342840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644344100,
          "schedule_time": 1644344100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644344220,
          "schedule_time": 1644344220,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644345180,
          "schedule_time": 1644345180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644345240,
          "schedule_time": 1644345240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:10011:6:11",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.818718,
            "lng": 8.774009
          }
        },
        "arrival": {
          "time": 1644346140,
          "schedule_time": 1644346140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644346140,
          "schedule_time": 1644346140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321001131",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.819252,
            "lng": 8.774304
          }
        },
        "arrival": {
          "time": 1644346260,
          "schedule_time": 1644346260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644346260,
          "schedule_time": 1644346260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.819545,
            "lng": 8.774208
          }
        },
        "arrival": {
          "time": 1644346320,
          "schedule_time": 1644346320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644346320,
          "schedule_time": 1644346320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644346740,
          "schedule_time": 1644346740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 4
          },
          "category_name": "Bus",
          "category_id": 0,
          "clasz": 10,
          "train_nr": 0,
          "line_id": "GB",
          "name": "Bus GB",
          "provider": "FS Omnibus GmbH & Co. KG",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 4,
            "to": 5
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 5,
            "to": 10
          },
          "category_name": "ICE",
          "category_id": 9,
          "clasz": 1,
          "train_nr": 1572,
          "line_id": "26",
          "name": "ICE 1572",
          "provider": "DB Fernverkehr AG",
          "direction": "Hannover Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 10,
            "to": 11
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 11,
            "to": 12
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 12,
            "to": 13
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 4
        },
        "id": {
          "station_id": "delfi_de:06432:16120:1:1",
          "train_nr": 93636,
          "time": 1644338100,
          "target_station_id": "delfi_de:06411:4734:49:49",
          "target_time": 1644340920,
          "line_id": "GB"
        },
      },
      {
        "range": {
          "from": 5,
          "to": 10
        },
        "id": {
          "station_id": "delfi_de:08212:90",
          "train_nr": 1572,
          "time": 1644336600,
          "target_station_id": "delfi_de:03241:31",
          "target_time": 1644353760,
          "line_id": "26"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  },
  {
    "stops": [
      {
        "station": {
          "id": "START",
          "name": "START",
          "pos": {
            "lat": 49.874955,
            "lng": 8.656523
          }
        },
        "arrival": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644340500,
          "schedule_time": 1644340500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4736:5:5",
          "name": "Darmstadt Luisenplatz",
          "pos": {
            "lat": 49.872913,
            "lng": 8.650806
          }
        },
        "arrival": {
          "time": 1644340980,
          "schedule_time": 1644340980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341100,
          "schedule_time": 1644341100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:24566:1:1",
          "name": "Darmstadt Rhein-/Neckarstraße",
          "pos": {
            "lat": 49.871902,
            "lng": 8.643649
          }
        },
        "arrival": {
          "time": 1644341220,
          "schedule_time": 1644341220,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341220,
          "schedule_time": 1644341220,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4735:3:3",
          "name": "Darmstadt Berliner Allee",
          "pos": {
            "lat": 49.870773,
            "lng": 8.635729
          }
        },
        "arrival": {
          "time": 1644341280,
          "schedule_time": 1644341280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341280,
          "schedule_time": 1644341280,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644341460,
          "schedule_time": 1644341460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341460,
          "schedule_time": 1644341460,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "arrival": {
          "time": 1644341580,
          "schedule_time": 1644341580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341580,
          "schedule_time": 1644341580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4734:64:63",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.873035,
            "lng": 8.629159
          }
        },
        "arrival": {
          "time": 1644341700,
          "schedule_time": 1644341700,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341700,
          "schedule_time": 1644341700,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06411:4733:1:1",
          "name": "Darmstadt-Arheilgen Bahnhof",
          "pos": {
            "lat": 49.913868,
            "lng": 8.645456
          }
        },
        "arrival": {
          "time": 1644341940,
          "schedule_time": 1644341940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644341940,
          "schedule_time": 1644341940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06411:4732:3:3",
          "name": "Darmstadt-Wixhausen Bahnhof",
          "pos": {
            "lat": 49.93034,
            "lng": 8.647816
          }
        },
        "arrival": {
          "time": 1644342060,
          "schedule_time": 1644342060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342060,
          "schedule_time": 1644342060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06432:4792:2:2",
          "name": "Erzhausen Bahnhof",
          "pos": {
            "lat": 49.951153,
            "lng": 8.650842
          }
        },
        "arrival": {
          "time": 1644342180,
          "schedule_time": 1644342180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342240,
          "schedule_time": 1644342240,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2706:1:1",
          "name": "Egelsbach Bahnhof",
          "pos": {
            "lat": 49.96817,
            "lng": 8.653729
          }
        },
        "arrival": {
          "time": 1644342360,
          "schedule_time": 1644342360,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342360,
          "schedule_time": 1644342360,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2705:2:2",
          "name": "Langen (Hessen) Bahnhof",
          "pos": {
            "lat": 49.993542,
            "lng": 8.656922
          }
        },
        "arrival": {
          "time": 1644342480,
          "schedule_time": 1644342480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342540,
          "schedule_time": 1644342540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:11505:1:1",
          "name": "Langen (Hessen) Flugsicherung",
          "pos": {
            "lat": 50.005276,
            "lng": 8.658583
          }
        },
        "arrival": {
          "time": 1644342600,
          "schedule_time": 1644342600,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342660,
          "schedule_time": 1644342660,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2702:2:2",
          "name": "Dreieich-Buchschlag Bahnhof",
          "pos": {
            "lat": 50.022583,
            "lng": 8.661405
          }
        },
        "arrival": {
          "time": 1644342780,
          "schedule_time": 1644342780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342780,
          "schedule_time": 1644342780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06438:2701:2:2",
          "name": "Neu-Isenburg Bahnhof",
          "pos": {
            "lat": 50.052856,
            "lng": 8.665492
          }
        },
        "arrival": {
          "time": 1644342960,
          "schedule_time": 1644342960,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644342960,
          "schedule_time": 1644342960,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:2799:11:1",
          "name": "Frankfurt (Main) Louisa Bahnhof",
          "pos": {
            "lat": 50.083672,
            "lng": 8.670308
          }
        },
        "arrival": {
          "time": 1644343140,
          "schedule_time": 1644343140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343140,
          "schedule_time": 1644343140,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:933:1:2",
          "name": "Frankfurt (Main) Stresemannallee Bahnhof",
          "pos": {
            "lat": 50.09454,
            "lng": 8.671268
          }
        },
        "arrival": {
          "time": 1644343260,
          "schedule_time": 1644343260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343260,
          "schedule_time": 1644343260,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:912:2:4",
          "name": "Frankfurt (Main) Südbahnhof",
          "pos": {
            "lat": 50.099026,
            "lng": 8.685507
          }
        },
        "arrival": {
          "time": 1644343320,
          "schedule_time": 1644343320,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343380,
          "schedule_time": 1644343380,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:907:11:12",
          "name": "Frankfurt (Main) Lokalbahnhof",
          "pos": {
            "lat": 50.101933,
            "lng": 8.692938
          }
        },
        "arrival": {
          "time": 1644343440,
          "schedule_time": 1644343440,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343500,
          "schedule_time": 1644343500,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:525:1:2",
          "name": "Frankfurt (Main) Ostendstraße",
          "pos": {
            "lat": 50.112385,
            "lng": 8.696968
          }
        },
        "arrival": {
          "time": 1644343560,
          "schedule_time": 1644343560,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343620,
          "schedule_time": 1644343620,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:510:1:3",
          "name": "Frankfurt (Main) Konstablerwache",
          "pos": {
            "lat": 50.114697,
            "lng": 8.685797
          }
        },
        "arrival": {
          "time": 1644343680,
          "schedule_time": 1644343680,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343740,
          "schedule_time": 1644343740,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:1:11:2",
          "name": "Frankfurt (Main) Hauptwache",
          "pos": {
            "lat": 50.114056,
            "lng": 8.678139
          }
        },
        "arrival": {
          "time": 1644343800,
          "schedule_time": 1644343800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343800,
          "schedule_time": 1644343800,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:11:1:2",
          "name": "Frankfurt (Main) Taunusanlage",
          "pos": {
            "lat": 50.113476,
            "lng": 8.668763
          }
        },
        "arrival": {
          "time": 1644343860,
          "schedule_time": 1644343860,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343920,
          "schedule_time": 1644343920,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:7010:2:1",
          "name": "Frankfurt (Main) Hauptbahnhof tief",
          "pos": {
            "lat": 50.107132,
            "lng": 8.662473
          }
        },
        "arrival": {
          "time": 1644343980,
          "schedule_time": 1644343980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644343980,
          "schedule_time": 1644343980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06412:10:18:22",
          "name": "Frankfurt (Main) Hauptbahnhof",
          "pos": {
            "lat": 50.106621,
            "lng": 8.662561
          }
        },
        "arrival": {
          "time": 1644344100,
          "schedule_time": 1644344100,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644344340,
          "schedule_time": 1644344340,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": true
      },
      {
        "station": {
          "id": "delfi_de:06440:6401:2:9",
          "name": "Friedberg (Hessen) Bahnhof",
          "pos": {
            "lat": 50.332035,
            "lng": 8.760561
          }
        },
        "arrival": {
          "time": 1644345840,
          "schedule_time": 1644345840,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644345900,
          "schedule_time": 1644345900,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06531:11016:12:12",
          "name": "Gießen Bahnhof",
          "pos": {
            "lat": 50.579884,
            "lng": 8.663799
          }
        },
        "arrival": {
          "time": 1644346980,
          "schedule_time": 1644346980,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644347040,
          "schedule_time": 1644347040,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_de:06534:10011:6:11",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.818718,
            "lng": 8.774009
          }
        },
        "arrival": {
          "time": 1644347940,
          "schedule_time": 1644347940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644347940,
          "schedule_time": 1644347940,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": true,
        "enter": false
      },
      {
        "station": {
          "id": "delfi_000321001131",
          "name": "Marburg Hauptbahnhof",
          "pos": {
            "lat": 50.819252,
            "lng": 8.774304
          }
        },
        "arrival": {
          "time": 1644348060,
          "schedule_time": 1644348060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644348060,
          "schedule_time": 1644348060,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "VIA0",
          "name": "VIA0",
          "pos": {
            "lat": 50.819545,
            "lng": 8.774208
          }
        },
        "arrival": {
          "time": 1644348120,
          "schedule_time": 1644348120,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 1644348120,
          "schedule_time": 1644348120,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      },
      {
        "station": {
          "id": "END",
          "name": "END",
          "pos": {
            "lat": 50.809807,
            "lng": 8.766264
          }
        },
        "arrival": {
          "time": 1644348540,
          "schedule_time": 1644348540,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        },
        "departure": {
          "time": 0,
          "schedule_time": 0,
          "track": "",
          "schedule_track": "",
          "valid": false,
          "reason": "SCHEDULE"
        },
        "exit": false,
        "enter": false
      }
    ],
    "transports": [
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 0,
            "to": 1
          },
          "mumo_id": 0,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 1,
            "to": 5
          },
          "category_name": "Str",
          "category_id": 1,
          "clasz": 9,
          "train_nr": 358,
          "line_id": "5",
          "name": "5",
          "provider": "HEAG Mobilo",
          "direction": "Darmstadt Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 5,
            "to": 6
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 6,
            "to": 23
          },
          "category_name": "S",
          "category_id": 7,
          "clasz": 7,
          "train_nr": 35362,
          "line_id": "S3",
          "name": "S3",
          "provider": "DB Regio AG S-Bahn Rhein-Main",
          "direction": "Bad Soden(Taunus)"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 23,
            "to": 24
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Transport",
        "move": {
          "range": {
            "from": 24,
            "to": 27
          },
          "category_name": "Regional Rail",
          "category_id": 11,
          "clasz": 6,
          "train_nr": 4164,
          "line_id": "RE30",
          "name": "RE30",
          "provider": "DB Regio AG Mitte Region Hessen",
          "direction": "Kassel Hauptbahnhof"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 27,
            "to": 28
          },
          "mumo_id": -1,
          "price": 0,
          "accessibility": 0,
          "mumo_type": ""
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 28,
            "to": 29
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "foot"
        }
      },
      {
        "move_type": "Walk",
        "move": {
          "range": {
            "from": 29,
            "to": 30
          },
          "mumo_id": 250023,
          "price": 0,
          "accessibility": 0,
          "mumo_type": "car"
        }
      }
    ],
    "trips": [
      {
        "range": {
          "from": 1,
          "to": 5
        },
        "id": {
          "station_id": "delfi_de:06411:11328:3:3",
          "train_nr": 358,
          "time": 1644340080,
          "target_station_id": "delfi_de:06411:4734:48:48",
          "target_time": 1644341580,
          "line_id": "5"
        },
      },
      {
        "range": {
          "from": 6,
          "to": 23
        },
        "id": {
          "station_id": "delfi_de:06411:4734:64:63",
          "train_nr": 35362,
          "time": 1644341700,
          "target_station_id": "delfi_de:06436:4299:1:2",
          "target_time": 1644345540,
          "line_id": "S3"
        },
      },
      {
        "range": {
          "from": 24,
          "to": 27
        },
        "id": {
          "station_id": "delfi_de:06412:10:18:22",
          "train_nr": 4164,
          "time": 1644344340,
          "target_station_id": "delfi_de:06611:200001",
          "target_time": 1644352440,
          "line_id": "RE30"
        },
      }
    ],
    "attributes": [],
    "free_texts": [],
    "problems": [],
    "night_penalty": 0,
    "db_costs": 0,
    "status": "OK"
  }
]
