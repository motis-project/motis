/* eslint-disable @typescript-eslint/no-explicit-any */
import { config, mount, VueWrapper } from "@vue/test-utils";
import flushPromises from "flush-promises";
import { mockNextAxiosPost } from "./TestHelpers";
import StationTimetable from "../../src/views/StationTimetable.vue";
import StationGuess from "../../src/models/StationGuess";

describe("TrainSearch test", () => {
  let wrapper: VueWrapper;

  beforeAll(() => {
    config.global.mocks = {
      $route: {
        name: "StationTimetable"
      }
    }
  })

  beforeEach(async () => {
    mockNextAxiosPost(stationGuessResponse);
    wrapper = mount(StationTimetable, {
      props: {
        stationGuess: stationGuessRequest
      }
    });
    await flushPromises();
  })

  it("Snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Elements from request rendered", () => {
    expect(wrapper.findAll('div .station-event')).toHaveLength(9);
    expect(wrapper.findAll('div .event-time')).toHaveLength(9);
    expect(wrapper.findAll('div .event-train')).toHaveLength(9);
    wrapper.findAll('input').map((i) => expect(i.attributes().type).toBe('radio'));
    expect(wrapper.findAll('div .extend-search-interval')).toHaveLength(2);
    expect(wrapper.findAll('.date-header')).toHaveLength(3);
    // loading bar not rendered
    expect(wrapper.findComponent("LoadingBar").exists()).toBeFalsy();
  })

  it("Ankunft/Abfahrt buttons work", async () => {
    wrapper.find('#station-arrivals').trigger('mousedown');
    wrapper.find('#station-arrivals').trigger('mouseup');
    await flushPromises();
    expect((wrapper.vm.$data as any).departures).toStrictEqual(stationGuessResponse.content.events);
    wrapper.find('#station-departures').trigger('mousedown');
    wrapper.find('#station-departures').trigger('mouseup');
    await flushPromises();
    expect((wrapper.vm.$data as any).departures).toStrictEqual(stationGuessResponse.content.events);
  })

  it("Frueher/Spaeter buttons work", async () => {
    const frueher = wrapper.find('div .search-before').find('button');
    const spaeter = wrapper.find('div .search-after').find('button');
    frueher.trigger('mousedown');
    frueher.trigger('mouseup');
    await flushPromises();
    expect((wrapper.vm.$data as any).departures).toStrictEqual(stationGuessResponse.content.events);
    spaeter.trigger('mousedown');
    spaeter.trigger('mouseup');
    await flushPromises();
    expect((wrapper.vm.$data as any).departures).toStrictEqual(stationGuessResponse.content.events);
  })
})


const stationGuessRequest: StationGuess = {
  id: "delfi_de:09278:3467:0:1",
  name: "Frath",
  pos: {
    lat: 49.019966,
    lng: 12.532181
  }
}


const stationGuessResponse = {
  "destination": {
    "type": "Module",
    "target": ""
  },
  "content_type": "RailVizStationResponse",
  "content": {
    "station": {
      "id": "delfi_de:09278:3467:0:1",
      "name": "Frath",
      "pos": {
        "lat": 49.019966,
        "lng": 12.532181
      }
    },
    "events": [
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646197440,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646201700,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646199780,
          "schedule_time": 1646199780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646197440,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646201700,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646199780,
          "schedule_time": 1646199780,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646204940,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646208120,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646206080,
          "schedule_time": 1646206080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646204940,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646208120,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646206080,
          "schedule_time": 1646206080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646225280,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646229000,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646227080,
          "schedule_time": 1646227080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646225280,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646229000,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646227080,
          "schedule_time": 1646227080,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646283840,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646288100,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646286180,
          "schedule_time": 1646286180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646283840,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646288100,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646286180,
          "schedule_time": 1646286180,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646291340,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646294520,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646292480,
          "schedule_time": 1646292480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646291340,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646294520,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646292480,
          "schedule_time": 1646292480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646311680,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646315400,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646313480,
          "schedule_time": 1646313480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646311680,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646315400,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646313480,
          "schedule_time": 1646313480,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646370240,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646374500,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646372580,
          "schedule_time": 1646372580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646370240,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646374500,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646372580,
          "schedule_time": 1646372580,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646377740,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646380920,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646378880,
          "schedule_time": 1646378880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3489:0:1",
              "train_nr": 0,
              "time": 1646377740,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646380920,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646378880,
          "schedule_time": 1646378880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646398080,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646401800,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "DEP",
        "event": {
          "time": 1646399880,
          "schedule_time": 1646399880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      },
      {
        "trips": [
          {
            "id": {
              "station_id": "delfi_de:09278:3482:0:1",
              "train_nr": 0,
              "time": 1646398080,
              "target_station_id": "delfi_de:09263:1924:0:1_G",
              "target_time": 1646401800,
              "line_id": "1005"
            },
            "transport": {
              "range": {
                "from": 0,
                "to": 0
              },
              "category_name": "Bus",
              "category_id": 0,
              "clasz": 10,
              "train_nr": 0,
              "line_id": "1005",
              "name": "Bus 1005",
              "provider": "Bus8",
              "direction": "Straubing Bf Bachstra\u00DFe"
            }
          }
        ],
        "type": "ARR",
        "event": {
          "time": 1646399880,
          "schedule_time": 1646399880,
          "track": "",
          "schedule_track": "",
          "valid": true,
          "reason": "SCHEDULE"
        }
      }
    ]
  },
  "id": 1
}
