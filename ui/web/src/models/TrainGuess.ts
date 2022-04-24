/* eslint-disable camelcase */
import StationGuess from "./StationGuess";
import TransportTrip from "./TransportTrip";

export interface TrainGuessResponseContent {
	trips: Trips[]
}

export interface TripInfoId {
	station_id: string,
	train_nr: number,
	time: number,
	target_station_id: string,
	target_time: number,
	line_id: string
}

export default interface Trips {
	first_station: StationGuess,
	trip_info: TransportTrip
}
