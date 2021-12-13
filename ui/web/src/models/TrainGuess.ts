import StationGuess from "./StationGuess";

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

export interface TripInfo {
	id: TripInfoId,
		transport: {
			range: {
				from: number,
				to: number,
			},
			category_name: string,
			category_id: number,
			clasz: number,
			train_number: number,
			line_id: string,
			name: string,
			provider: string,
			direction: string
		}
}

export default interface Trips {
	first_station: StationGuess,
	trip_info: TripInfo
}
