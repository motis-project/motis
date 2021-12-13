import axios from 'axios'
import { App } from 'vue'
import StationGuess, { StationGuessResponseContent } from '../models/StationGuess';
import { AddressGuessResponseContent } from '../models/AddressGuess';
import Trip from '../models/Trip';
import TripResponseContent from '../models/TripResponseContent';
import { RailVizStationResponseContent } from '../models/DepartureTimetable';
import { stringifyQuery } from 'vue-router';
import { TrainGuessResponseContent } from '@/models/TrainGuess'


var service: MOTISPostService = {
  async getStationGuessResponse(input: string, gc: number) {
    let rq = {
      destination: {
        type: "Module",
        target: "/guesser"
      },
      content_type: "StationGuesserRequest",
      content: {
        input: input,
        guess_count: gc
      }
    };
    return (await axios.post<StationGuessResponse>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getAddressGuessResponse(input: string) {
    let rq = {
      destination: {
        type: "Module",
        target: "/address"
      },
      content_type: "AddressRequest",
      content: {
        input: input
      }
    };
    return (await axios.post<AddressGuessResponse>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getTripResponce(trip: Trip) {
    let rq = {
      destination: {
        type: "Module",
        target: "/trip_to_connection"
      },
      content_type: "TripId",
      content: trip
    };
    return (await axios.post<TripResponce>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getDeparturesResponse(station: string, by_schedule_time: Boolean, direction: string, event_count: number, time: number) {
    let rq = {
      destination: {
        type: "Module",
        target: "/railviz/get_station"
      },
      content_type: "RailVizStationRequest",
      content: {
        by_schedule_time: by_schedule_time,
        direction: direction,
        event_count: event_count,
        station_id: station,
        time: time
      },
    };
  return (await axios.post<RailVizStationResponse>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getTrainGuessResponse(currentTime: number, currentTrainNum: number){
    let rq = {
        destination: {
            target: "/railviz/get_trip_guesses",
            type: "Module"
        },
        content_type: "RailVizTripGuessRequest",
        content: {
            guess_count: 20,
            time: currentTime,
            train_num: currentTrainNum
        },
    }
    return (await axios.post<TrainGuessResponse>("https://europe.motis-project.de/", rq)).data.content;
  }
}

interface StationGuessResponse {
  destination: {
    type: string,
    target: string
  },
  content_type: string,
  content: StationGuessResponseContent,
  id: number
}

interface AddressGuessResponse {
  destination: {
    target: string
  },
  content_type: string,
  content: AddressGuessResponseContent,
  id: number
}

interface TripResponce {
  destination: {
    type: string,
    target: string
  },
  content_type: string,
  content: TripResponseContent,
  id: number
}

interface RailVizStationResponse {
  destination: {
    type: string,
    target: string
  },
  content_type: string,
  content: RailVizStationResponseContent,
  id: number
}

interface TrainGuessResponse {
  destination: {
      type: string,
      target: string
  },
  content_type: string,
  content: TrainGuessResponseContent,
  id: 1
}


interface MOTISPostService {
  getStationGuessResponse(input: string, gc: number): Promise<StationGuessResponseContent>
  getAddressGuessResponse(input: string): Promise<AddressGuessResponseContent>
  getTripResponce(input: Trip) : Promise<TripResponseContent>
  getDeparturesResponse(station: string, by_schedule_time: Boolean, direction: string, event_count: number, time: number) : Promise<RailVizStationResponseContent>
  getTrainGuessResponse(currentTime: number, currentTrainNum: number): Promise<TrainGuessResponseContent>
}


















declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $postService: MOTISPostService
  }
}

export default {
  install: (app: App, options: string) => {
    app.config.globalProperties.$postService = service;
  }
}