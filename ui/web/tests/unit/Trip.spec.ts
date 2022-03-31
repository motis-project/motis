/* eslint-disable @typescript-eslint/no-explicit-any */
import { config, mount, VueWrapper } from "@vue/test-utils";
import { mockNextAxiosPost } from "./TestHelpers";
import Trip from "../../src/views/Trip.vue"
import TripResponseContent from "../../src/models/TripResponseContent"
import TripModel from "../../src/models/Trip";
import flushPromises from "flush-promises";

describe('Test Trip.vue with single connection', () => {
  let wrapper: VueWrapper;

  beforeAll(() => {
    config.global.mocks = {
      $route: {
        name: "Trip"
      }
    }
  })

  beforeEach(async () => {
    mockNextAxiosPost(mockTripResponseContent);
    wrapper = mount(Trip, {
      props: {
        trip: mockTrip,
      }
    })
    await flushPromises();
  })

  it("Snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Loads content", () => {
    expect((wrapper.vm.$data as any).content).toStrictEqual(mockTripResponseContent.content);
    expect((wrapper.vm.$data as any).isContentLoaded).toBe(true);
  })

  it("Expand button works", async () => {
    const button = wrapper.get(".intermediate-stops-toggle");
    const list = wrapper.get(".intermediate-stops");
    expect(list.classes()).toContain("expanded");
    button.trigger("click");
    await flushPromises();
    expect(list.classes()).not.toContain("expanded");
    button.trigger("click");
    await flushPromises();
    expect(list.classes()).toContain("expanded");
  })
});




//#region Mocked values

/* eslint-disable camelcase */
const mockTrip: TripModel = {
  "station_id": "delfi_de:06411:4734:64:63",
  "train_nr": 35354,
  "time": 1642952100,
  "target_station_id": "delfi_de:06436:4299:1:2",
  "target_time": 1642955940,
  "line_id": "S3"
}

const mockTripResponseContent = {
  "destination": {
      "type": "Module",
      "target": ""
  },
  "content_type": "Connection",
  "content": {
      "stops": [
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
                  "time": 0,
                  "schedule_time": 0,
                  "track": "",
                  "schedule_track": "",
                  "valid": false,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644321900,
                  "schedule_time": 1644321900,
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
                  "time": 1644322140,
                  "schedule_time": 1644322140,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322140,
                  "schedule_time": 1644322140,
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
                  "time": 1644322260,
                  "schedule_time": 1644322260,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322260,
                  "schedule_time": 1644322260,
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
                  "time": 1644322380,
                  "schedule_time": 1644322380,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322440,
                  "schedule_time": 1644322440,
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
                  "time": 1644322560,
                  "schedule_time": 1644322560,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322560,
                  "schedule_time": 1644322560,
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
                  "id": "delfi_de:06438:2705:2:3",
                  "name": "Langen (Hessen) Bahnhof",
                  "pos": {
                      "lat": 49.993542,
                      "lng": 8.65688
                  }
              },
              "arrival": {
                  "time": 1644322680,
                  "schedule_time": 1644322680,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322740,
                  "schedule_time": 1644322740,
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
                  "time": 1644322800,
                  "schedule_time": 1644322800,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322860,
                  "schedule_time": 1644322860,
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
                  "time": 1644322980,
                  "schedule_time": 1644322980,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644322980,
                  "schedule_time": 1644322980,
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
                  "time": 1644323160,
                  "schedule_time": 1644323160,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323160,
                  "schedule_time": 1644323160,
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
                  "time": 1644323340,
                  "schedule_time": 1644323340,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323340,
                  "schedule_time": 1644323340,
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
                  "time": 1644323460,
                  "schedule_time": 1644323460,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323460,
                  "schedule_time": 1644323460,
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
                  "time": 1644323520,
                  "schedule_time": 1644323520,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323580,
                  "schedule_time": 1644323580,
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
                  "time": 1644323640,
                  "schedule_time": 1644323640,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323700,
                  "schedule_time": 1644323700,
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
                  "time": 1644323760,
                  "schedule_time": 1644323760,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323820,
                  "schedule_time": 1644323820,
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
                  "time": 1644323880,
                  "schedule_time": 1644323880,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644323940,
                  "schedule_time": 1644323940,
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
                  "time": 1644324000,
                  "schedule_time": 1644324000,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324000,
                  "schedule_time": 1644324000,
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
                  "time": 1644324060,
                  "schedule_time": 1644324060,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324120,
                  "schedule_time": 1644324120,
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
                  "time": 1644324180,
                  "schedule_time": 1644324180,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324240,
                  "schedule_time": 1644324240,
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
                  "time": 1644324360,
                  "schedule_time": 1644324360,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324420,
                  "schedule_time": 1644324420,
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
                  "time": 1644324480,
                  "schedule_time": 1644324480,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324480,
                  "schedule_time": 1644324480,
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
                  "time": 1644324540,
                  "schedule_time": 1644324540,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324600,
                  "schedule_time": 1644324600,
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
                  "id": "delfi_de:06412:1217:1:1",
                  "name": "Frankfurt (Main) Rödelheim Bahnhof",
                  "pos": {
                      "lat": 50.125381,
                      "lng": 8.606896
                  }
              },
              "arrival": {
                  "time": 1644324780,
                  "schedule_time": 1644324780,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324780,
                  "schedule_time": 1644324780,
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
                  "id": "delfi_de:06436:2299:2:2",
                  "name": "Eschborn Südbahnhof",
                  "pos": {
                      "lat": 50.134033,
                      "lng": 8.577355
                  }
              },
              "arrival": {
                  "time": 1644324960,
                  "schedule_time": 1644324960,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644324960,
                  "schedule_time": 1644324960,
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
                  "id": "delfi_de:06436:2201:1:2",
                  "name": "Eschborn Bahnhof",
                  "pos": {
                      "lat": 50.1436,
                      "lng": 8.561045
                  }
              },
              "arrival": {
                  "time": 1644325080,
                  "schedule_time": 1644325080,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644325080,
                  "schedule_time": 1644325080,
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
                  "id": "delfi_de:06436:2213:1:3",
                  "name": "Eschborn-Niederhöchstadt Bahnhof",
                  "pos": {
                      "lat": 50.154263,
                      "lng": 8.547066
                  }
              },
              "arrival": {
                  "time": 1644325200,
                  "schedule_time": 1644325200,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644325260,
                  "schedule_time": 1644325260,
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
                  "id": "delfi_de:06436:11106:1:1",
                  "name": "Schwalbach (Taunus) Nord",
                  "pos": {
                      "lat": 50.159786,
                      "lng": 8.5346
                  }
              },
              "arrival": {
                  "time": 1644325380,
                  "schedule_time": 1644325380,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644325380,
                  "schedule_time": 1644325380,
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
                  "id": "delfi_de:06436:2243:1:1",
                  "name": "Schwalbach (Taunus) Limes Bahnhof",
                  "pos": {
                      "lat": 50.154583,
                      "lng": 8.528215
                  }
              },
              "arrival": {
                  "time": 1644325440,
                  "schedule_time": 1644325440,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644325500,
                  "schedule_time": 1644325500,
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
                  "id": "delfi_de:06436:2231:1:1",
                  "name": "Sulzbach (Taunus) Nordbahnhof",
                  "pos": {
                      "lat": 50.13958,
                      "lng": 8.518208
                  }
              },
              "arrival": {
                  "time": 1644325620,
                  "schedule_time": 1644325620,
                  "track": "",
                  "schedule_track": "",
                  "valid": true,
                  "reason": "SCHEDULE"
              },
              "departure": {
                  "time": 1644325620,
                  "schedule_time": 1644325620,
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
                  "id": "delfi_de:06436:4299:1:2",
                  "name": "Bad Soden (Taunus) Bahnhof",
                  "pos": {
                      "lat": 50.142853,
                      "lng": 8.504547
                  }
              },
              "arrival": {
                  "time": 1644325740,
                  "schedule_time": 1644325740,
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
              "exit": true,
              "enter": false
          }
      ],
      "transports": [
          {
              "move_type": "Transport",
              "move": {
                  "range": {
                      "from": 0,
                      "to": 28
                  },
                  "category_name": "S",
                  "category_id": 7,
                  "clasz": 7,
                  "train_nr": 35340,
                  "line_id": "S3",
                  "name": "S3",
                  "provider": "DB Regio AG S-Bahn Rhein-Main",
                  "direction": "Bad Soden(Taunus)"
              }
          }
      ],
      "trips": [
          {
              "range": {
                  "from": 0,
                  "to": 28
              },
              "id": {
                  "station_id": "delfi_de:06411:4734:64:63",
                  "train_nr": 35340,
                  "time": 1644321900,
                  "target_station_id": "delfi_de:06436:4299:1:2",
                  "target_time": 1644325740,
                  "line_id": "S3"
              },
          }
      ],
      "attributes": [],
      "free_texts": [],
      "problems": [],
      "night_penalty": 0,
      "db_costs": 0,
      "status": "OK"
  } as TripResponseContent,
  "id": 1
}

//#endregion mocked values
