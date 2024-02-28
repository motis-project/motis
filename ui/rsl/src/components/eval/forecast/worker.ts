import { ServiceClass } from "@/api/constants.ts";
import { setApiEndpoint } from "@/api/endpoint.ts";
import { sendLookupScheduleInfoRequest } from "@/api/lookup.ts";
import {
  sendPaxMonCheckDataRequest,
  sendPaxMonFilterTripsRequest,
  sendPaxMonGetTripLoadInfosRequest,
  sendPaxMonStatusRequest,
} from "@/api/paxmon.ts";

import { formatISODateTime } from "@/util/dateFormat.ts";

import {
  EvalResult,
  TripEvalResult,
  TripEvalSectionInfo,
  WorkerRequest,
  WorkerUpdate,
} from "@/components/eval/forecast/workerMessages.ts";

function sqr(x: number) {
  return x * x;
}

async function evaluation() {
  const scheduleInfo = await sendLookupScheduleInfoRequest();
  const paxmonStatus = await sendPaxMonStatusRequest({ universe: 0 });

  const intervalStart = scheduleInfo.begin;
  const intervalEnd = paxmonStatus.system_time;

  const tripList = await sendPaxMonFilterTripsRequest({
    universe: 0,
    ignore_past_sections: false,
    include_load_threshold: 0.0,
    critical_load_threshold: 1.0,
    crowded_load_threshold: 0.8,
    include_edges: true,
    sort_by: "TrainNr",
    max_results: 0,
    skip_first: 0,
    filter_by_time: "ActiveTime",
    filter_interval: {
      begin: intervalStart,
      end: intervalEnd,
    },
    filter_by_train_nr: false,
    filter_train_nrs: [],
    filter_by_service_class: true,
    filter_service_classes: [ServiceClass.ICE, ServiceClass.IC],
    filter_by_capacity_status: true,
    filter_has_trip_formation: true,
    filter_has_capacity_for_all_sections: true,
  });

  const totalTrips = tripList.filtered_trips;
  let evaluatedTrips = 0;

  postMessage({
    type: "TripCount",
    totalTrips,
  } as WorkerUpdate);

  const trips: TripEvalResult[] = [];
  let progress = 0;
  let totalQ50Mae = 0;
  let totalQ50Mse = 0;

  for (const trip of tripList.trips) {
    progress++;
    const checkData = await sendPaxMonCheckDataRequest({
      trip_id: trip.tsi.trip,
    });
    if (checkData.matched_entry_count == 0) {
      continue;
    }
    const loadInfos = await sendPaxMonGetTripLoadInfosRequest({
      universe: 0,
      trips: [trip.tsi.trip],
    });
    const loadData = loadInfos.load_infos[0];

    if (checkData.sections.length != loadData.edges.length) {
      console.log(
        `Auslastungsdaten und Zähldaten passen nicht zusammen bei Trip ${JSON.stringify(trip.tsi.trip)}.`,
      );
      continue;
    }

    const sections: TripEvalSectionInfo[] = [];
    let evaluatedSectionCount = 0;
    let deviationAvg = 0;
    let q50Mae = 0;
    let q50Mse = 0;
    let expectedMae = 0;
    let expectedMse = 0;

    for (let i = 0; i < checkData.sections.length; i++) {
      const cd = checkData.sections[i];
      const ld = loadData.edges[i];

      const duration =
        (cd.arrival_current_time - cd.departure_current_time) / 60;

      if (cd.check_count < 5 || duration < 10) {
        continue;
      }

      const checkPaxMin = cd.min_pax_count;
      const checkPaxMax = cd.max_pax_count;
      const checkPaxAvg = cd.avg_pax_count;

      const expectedPax = ld.expected_passengers;
      const forecastPaxQ5 = ld.dist.q5;
      const forecastPaxQ50 = ld.dist.q50;
      const forecastPaxQ95 = ld.dist.q95;

      const q50Diff = Math.abs(forecastPaxQ50 - checkPaxAvg);
      const q50Factor = forecastPaxQ50 / checkPaxAvg;
      const expectedDiff = Math.abs(expectedPax - checkPaxAvg);
      const expectedFactor = expectedPax / checkPaxAvg;
      const deviation = Math.abs(forecastPaxQ50 - expectedPax);

      deviationAvg += deviation;

      q50Mae += q50Diff;
      q50Mse += sqr(q50Diff);

      expectedMae += expectedDiff;
      expectedMse += sqr(expectedDiff);

      evaluatedSectionCount++;
      sections.push({
        from: cd.from,
        to: cd.to,
        departureScheduleTime: cd.departure_schedule_time,
        departureCurrentTime: cd.departure_current_time,
        arrivalScheduleTime: cd.arrival_schedule_time,
        arrivalCurrentTime: cd.arrival_current_time,
        duration,
        checkCount: cd.check_count,
        checkPaxMin,
        checkPaxMax,
        checkPaxAvg,
        checkSpread: checkPaxMax - checkPaxMin,
        expectedPax,
        forecastPaxQ5,
        forecastPaxQ50,
        forecastPaxQ95,
        forecastSpread: forecastPaxQ95 - forecastPaxQ5,
        deviation,
        q50Diff,
        q50Factor,
        expectedDiff,
        expectedFactor,
      });
    }

    if (evaluatedSectionCount == 0) {
      continue;
    }

    deviationAvg /= evaluatedSectionCount;
    q50Mae /= evaluatedSectionCount;
    q50Mse /= evaluatedSectionCount;
    expectedMae /= evaluatedSectionCount;
    expectedMse /= evaluatedSectionCount;

    const result: TripEvalResult = {
      tsi: trip.tsi,
      totalSectionCount: loadData.edges.length,
      evaluatedSectionCount,
      sections,
      deviationAvg,
      q50Mae,
      q50Mse,
      expectedMae,
      expectedMse,
    };

    totalQ50Mae += q50Mae;
    totalQ50Mse += q50Mse;

    evaluatedTrips++;
    trips.push(result);
    postMessage({
      type: "TripInfo",
      progress,
      result,
    } as WorkerUpdate);
  }

  totalQ50Mae /= evaluatedTrips;
  totalQ50Mse /= evaluatedTrips;

  const result: EvalResult = {
    trips,
    intervalStart,
    intervalEnd,
    q50Mae: totalQ50Mae,
    q50Mse: totalQ50Mse,
    tripCsv: createTripCsv(trips),
    sectionCsv: createSectionCsv(trips),
  };

  postMessage({
    type: "Done",
    progress,
    totalTrips,
    evaluatedTrips,
    result,
  } as WorkerUpdate);
}

function createTripCsv(trips: TripEvalResult[]): string {
  let csv =
    [
      "category",
      "train_nr",
      "train_start_eva",
      "train_start_name",
      "train_start_time",
      "train_end_eva",
      "train_end_name",
      "train_end_time",
      "total_sections",
      "evaluated_sections",
      "deviation_avg",
      "q50_mae",
      "q50_mse",
      "expected_mae",
      "expected_mse",
    ].join(",") + "\n";
  for (const trip of trips) {
    const si = trip.tsi.service_infos[0];
    const category = si?.category ?? "";
    csv +=
      [
        JSON.stringify(category),
        trip.tsi.trip.train_nr,
        trip.tsi.primary_station.id,
        JSON.stringify(trip.tsi.primary_station.name),
        formatISODateTime(trip.tsi.trip.time),
        trip.tsi.secondary_station.id,
        JSON.stringify(trip.tsi.secondary_station.name),
        formatISODateTime(trip.tsi.trip.target_time),
        trip.totalSectionCount,
        trip.evaluatedSectionCount,
        trip.deviationAvg,
        trip.q50Mae,
        trip.q50Mse,
        trip.expectedMae,
        trip.expectedMse,
      ].join(",") + "\n";
  }
  return csv;
}

function createSectionCsv(trips: TripEvalResult[]): string {
  let csv =
    [
      "category",
      "train_nr",

      "section_from_eva",
      "section_from_name",
      "section_from_schedule_time",
      "section_from_real_time",
      "section_to_eva",
      "section_to_name",
      "section_to_schedule_time",
      "section_to_real_time",
      "section_duration",

      "check_count",
      "check_pax_min",
      "check_pax_max",
      "check_pax_avg",
      "check_pax_spread",

      "expected_pax",
      "forecast_pax_q5",
      "forecast_pax_q50",
      "forecast_pax_q95",
      "forecast_spread",
      "deviation",

      "q50_diff",
      "q50_factor",
      "expected_diff",
      "expected_factor",

      "train_start_eva",
      "train_start_name",
      "train_start_time",
      "train_end_eva",
      "train_end_name",
      "train_end_time",
    ].join(",") + "\n";
  for (const trip of trips) {
    const si = trip.tsi.service_infos[0];
    const category = si?.category ?? "";

    const prefix =
      [JSON.stringify(category), trip.tsi.trip.train_nr].join(",") + ",";

    const suffix =
      [
        trip.tsi.primary_station.id,
        JSON.stringify(trip.tsi.primary_station.name),
        formatISODateTime(trip.tsi.trip.time),
        trip.tsi.secondary_station.id,
        JSON.stringify(trip.tsi.secondary_station.name),
        formatISODateTime(trip.tsi.trip.target_time),
      ].join(",") + "\n";

    for (const section of trip.sections) {
      csv +=
        prefix +
        [
          section.from.id,
          JSON.stringify(section.from.name),
          formatISODateTime(section.departureScheduleTime),
          formatISODateTime(section.departureCurrentTime),
          section.to.id,
          JSON.stringify(section.to.name),
          formatISODateTime(section.arrivalScheduleTime),
          formatISODateTime(section.arrivalCurrentTime),
          section.duration,

          section.checkCount,
          section.checkPaxMin,
          section.checkPaxMax,
          section.checkPaxAvg,
          section.checkSpread,

          section.expectedPax,
          section.forecastPaxQ5,
          section.forecastPaxQ50,
          section.forecastPaxQ95,
          section.forecastSpread,
          section.deviation,

          section.q50Diff,
          section.q50Factor,
          section.expectedDiff,
          section.expectedFactor,
        ].join(",") +
        "," +
        suffix;
    }
  }
  return csv;
}

onmessage = async (msg) => {
  const req = msg.data as WorkerRequest;

  switch (req.action) {
    case "Start": {
      setApiEndpoint(req.apiEndpoint);
      await evaluation();
      break;
    }
  }
};
